import openai
from typing import Optional, Tuple
import yaml

import config

from generate_sql_prompts import Prompts
from data.db import execute_query


class TextToSQLGenerator:
    """Генератор SQL запросов из текстовых описаний с использованием LLM."""
    
    def __init__(self, client: openai.OpenAI, schema_yaml_path: str, model: str = "gpt-4o"):
        """
        Args:
            client: Авторизованный клиент OpenAI
            schema_yaml_path: Путь к YAML файлу со схемой БД
            model: Модель для использования (gpt-4o, gpt-4o-mini, o1-preview)
        """
        self.client = client
        self.model = model
        
        # Загружаем схему из YAML
        with open(schema_yaml_path, 'r', encoding='utf-8') as f:
            self.schema = yaml.safe_load(f)
        
        # Формируем системный промпт
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Создает системный промпт с описанием схемы БД."""
        schema_yaml = yaml.dump(self.schema, allow_unicode=True, sort_keys=False)
        
        return Prompts.init_system.format(schema_yaml=schema_yaml)

    def _create_error_feedback(self, sql_query: str, error_message: str, attempt: int) -> str:
        """Создает feedback сообщение для LLM с описанием ошибки."""
        return Prompts.feedback_loop.format(sql_query=sql_query, error_message=error_message, attempt=attempt)


    def _clean_sql_output(self, sql: str) -> str:
        """Удаляет markdown форматирование из SQL."""
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip()
    

    def generate_sql(self, user_query: str, temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Генерирует SQL запрос на основе текстового описания.
        
        Args:
            user_query: Текстовый запрос пользователя
            temperature: Температура генерации (0.0-1.0). Низкая для детерминированности
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Строка с SQL запросом
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            sql_query = self._clean_sql_output(sql_query)
            
            return sql_query
            
        except Exception as e:
            raise Exception(f"Ошибка при генерации SQL: {str(e)}")
    
    def generate_sql_with_retry(
        self, 
        user_query: str,
        duckdb_connection,
        max_retries: int = 3,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[str], int]:
        """
        Генерирует SQL с автоматической коррекцией ошибок через feedback loop.
        
        Этот метод реализует Evaluator-Optimizer Pattern:
        1. Генерирует SQL запрос
        2. Валидирует его через EXPLAIN в DuckDB
        3. Если есть ошибка - передает ее обратно в LLM для исправления
        4. Повторяет до max_retries раз
        
        Args:
            user_query: Текстовый запрос пользователя
            duckdb_connection: Соединение с DuckDB для валидации
            max_retries: Максимальное количество попыток исправления (по умолчанию 3)
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            verbose: Выводить логи процесса исправления
            
        Returns:
            Кортеж (sql_query, error_message, attempts_count):
            - sql_query: Финальный SQL запрос или None если не удалось
            - error_message: Сообщение об ошибке или None если успех
            - attempts_count: Количество затраченных попыток
        """
        # История диалога для контекста
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        for attempt in range(1, max_retries + 1):
            try:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Попытка {attempt}/{max_retries}")
                    print(f"{'='*60}")
                
                # Генерируем SQL через API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                sql_query = response.choices[0].message.content.strip()
                sql_query = self._clean_sql_output(sql_query)
                
                if verbose:
                    print(f"\nСгенерированный SQL:\n{sql_query}\n")
                
                # Добавляем ответ LLM в историю
                messages.append({
                    "role": "assistant",
                    "content": sql_query
                })
                
                # Валидация через EXPLAIN
                try:
                    duckdb_connection.execute(f"EXPLAIN {sql_query}")
                    
                    if verbose:
                        print("✅ SQL валидация успешна!")
                    
                    # Успех! Возвращаем результат
                    return sql_query, None
                    
                except Exception as db_error:
                    error_message = str(db_error)
                    
                    if verbose:
                        print(f"❌ Ошибка валидации: {error_message}")
                    
                    # Если это последняя попытка - возвращаем ошибку
                    if attempt == max_retries:
                        if verbose:
                            print(f"\n⚠️ Достигнут лимит попыток ({max_retries})")
                        return sql_query, f"SQL ошибка после {max_retries} попыток: {error_message}"
                    
                    # Формируем feedback для LLM
                    feedback_message = self._create_error_feedback( sql_query, error_message, attempt)
                    
                    if verbose:
                        print(f"\n🔄 Отправляю feedback в LLM для исправления...")
                    
                    # Добавляем feedback в историю для следующей итерации
                    messages.append({
                        "role": "user",
                        "content": feedback_message
                    })
                    
            except Exception as e:
                # Ошибка при вызове OpenAI API
                return None, f"Ошибка API на попытке {attempt}: {str(e)}"
        
        # Этот код не должен выполниться, но на всякий случай
        return None, "Неожиданная ошибка в цикле retry"

def text2df(
    text_request: str,
    db_con
): 
    """
    Принимает текстовый пользовательский запрос и возвращает DataFrame с необходимыми данными.
    
    :param text_request - str: Свалидированный текстовый пользовательский запрос
    :param db_con: Коннектор к DuckDB
    """

    global config

    client = openai.OpenAI(
        api_key=config.API_KEY,
        base_url="https://llm.api.cloud.yandex.net/v1",
        project=config.FOLDER_ID
    )

    generator = TextToSQLGenerator(
        client=client,
        schema_yaml_path='data/schema.py',
        model=config.SQL_GEN_MODEL
    )

    sql_query, error = generator.generate_sql_with_retry(text_request, db_con, verbose=True)
    if error is not None:
        raise RuntimeError(error)

    df = execute_query(db_con, sql_query)
    return df
