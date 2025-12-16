import json
import pandas as pd
import duckdb

def get_db_con(data_path):
    with open(data_path, 'r') as f:
        _json = json.load(f)
    
    raw_df = pd.json_normalize(_json, sep='_').drop(columns=['id'])
    raw_df.columns = raw_df.columns.str.replace('data_', '')
    
    raw_df['is_one_day_offer_available'] = ~raw_df['one_day_offer_content_version'].isna()
    raw_df['is_one_day_offer_v3_available'] = ~raw_df['one_day_offer_content_v3_date'].isna()
    
    raw_df = raw_df.drop(columns=['english_level', 'one_day_offer_content', 'one_day_offer_content_v3', 'is_my', 'application', 'is_form_my_company'])
    raw_df = raw_df.drop(columns=raw_df.columns[raw_df.columns.str.contains(r'^one\_day\_offer.*', regex=True)])
    
    multiple_choise_columns = {
        'locations': {'column_name': 'location', 'table_name': 'Locations'}, 
        'stack': {'column_name': 'skill', 'table_name': 'Skills'}, 
        'breadcrumbs': {'column_name': 'breadcrumb', 'table_name': 'Breadcrumbs'}, 
        'specializations': {'column_name': 'specialization', 'table_name': 'Specializations'}, 
        'relocation_options': {'column_name': 'relocation_option', 'table_name': 'RelocationOptions'}, 
        'display_locations': {'column_name': 'display_location', 'table_name': 'DisplayLocation', 'process_as_table': True}
    }
    
    external_tables = {}
    for column, params in multiple_choise_columns.items():
        feature_link = raw_df.set_index('id')[column].explode().dropna().rename(params['column_name'])
        if params.get('process_as_table', False):
            feature_link = pd.DataFrame(feature_link.to_dict()).T
        feature_link.index.name = 'vacancy_id'
        external_tables[params['table_name']] = feature_link.reset_index()
    
    raw_df = (
        raw_df
        .drop(columns=multiple_choise_columns.keys())
        .rename(columns={'id': 'vacancy_id'})
    )
    
    # Подключение к in-memory базе (или укажите путь к файлу: 'my_database.duckdb')
    con = duckdb.connect("data/vacancies.duckdb")
    
    # Регистрация основного DataFrame (замените 'main_df' на ваше имя)
    con.register('Vacancies', raw_df)
    
    # Регистрация 6 небольших датасетов (замените имена на ваши)
    for name, _df in external_tables.items():
        con.register(name, _df)
    
    # Например средние зарплаты вакансий из топ 100 по попурярности позиций Москвы
    return con
def execute_query(con, query):
    return con.execute(query).df()
