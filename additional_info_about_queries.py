from typing import List
import re
import numpy as np
import pandas as pd


def top3_facts(query: str, df: pd.DataFrame) -> List[str]:
    def _clean(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _tok(s: str):
        s = (s or "").lower()
        s = re.sub(r"[^a-zа-яё0-9_]+", " ", s)
        return [t for t in s.split() if t]

    def _col_tokens(col: str):
        s = (col or "").lower()
        s = re.sub(r"[^a-zа-яё0-9]+", " ", s)
        parts = []
        for p in s.split():
            parts += p.split("_")
        return [p for p in parts if p]

    def _relevance(qt, col: str) -> float:
        ct = set(_col_tokens(col))
        if not ct:
            return 0.0
        inter = len(set(qt) & ct)
        pref = 0
        for q in set(qt):
            for c in ct:
                if len(q) >= 4 and (c.startswith(q) or q.startswith(c)):
                    pref += 1
        return inter * 2.0 + min(pref, 3) * 0.7

    def _find_datetime_cols(d: pd.DataFrame):
        dt_cols = []
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                dt_cols.append(c)
            elif re.search(r"(date|dt|time|created|published|updated)", c.lower()):
                parsed = pd.to_datetime(d[c], errors="coerce")
                if parsed.notna().mean() >= 0.6:
                    dt_cols.append(c)
        return dt_cols

    def _cand(text: str, base: float, surprise: float = 0.0, rel: float = 0.0):
        return {"text": text, "score": base + surprise + rel}

    q = _clean(query)
    if df is None:
        return ["данные не переданы.", "нельзя построить факты без df.", "проверь этап получения данных."]

    if df.empty:
        return [
            f"по запросу «{q}» результат пустой: 0 строк.",
            "вероятно, фильтры слишком жёсткие или нет совпадений в данных.",
            "попробуй ослабить условия (период/локация/уровень/специализация).",
        ]

    qt = _tok(q)
    n = len(df)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    dt_cols = _find_datetime_cols(df)
    cat_cols = [c for c in df.columns if c not in num_cols and c not in dt_cols]

    cand = []

    # size
    sp = 2.2 if n < 30 else (1.2 if n < 200 else (1.8 if n > 50000 else 0.0))
    cand.append(_cand(f"размер выборки: {n} строк.", base=1.0, surprise=sp))

    # period + peak month
    if dt_cols:
        dt_col = sorted(dt_cols, key=lambda c: -_relevance(qt, c))[0]
        dt = pd.to_datetime(df[dt_col], errors="coerce")
        if dt.notna().any():
            dmin, dmax = dt.min(), dt.max()
            span_days = (dmax - dmin).days
            sp = 1.6 if span_days <= 7 else (1.2 if span_days >= 730 else 0.0)
            cand.append(_cand(
                f"период данных: {dmin.date()} — {dmax.date()} (колонка {dt_col}).",
                base=1.1, surprise=sp, rel=_relevance(qt, dt_col)
            ))
            m = dt.dt.to_period("M").astype(str)
            vc = m.value_counts()
            if len(vc) >= 2:
                top_m, top_cnt = vc.index[0], int(vc.iloc[0])
                share = top_cnt / n
                sp2 = 1.6 if share >= 0.5 else (0.9 if share >= 0.35 else 0.0)
                cand.append(_cand(
                    f"пик публикаций: {top_m} — {top_cnt} строк ({share*100:.1f}%).",
                    base=0.9, surprise=sp2, rel=_relevance(qt, dt_col)
                ))

    # missing
    miss = df.isna().mean().sort_values(ascending=False)
    miss_top = miss[miss > 0.0].head(1)
    if len(miss_top) > 0:
        c0, r0 = miss_top.index[0], float(miss_top.iloc[0])
        if r0 >= 0.10:
            sp = 2.0 if r0 >= 0.5 else (1.2 if r0 >= 0.25 else 0.6)
            cand.append(_cand(
                f"качество данных: высокая доля пропусков в «{c0}» — {r0*100:.1f}%.",
                base=0.9, surprise=sp, rel=_relevance(qt, c0)
            ))

    # duplicates
    dup = float(df.duplicated().mean())
    if dup >= 0.01:
        cand.append(_cand(
            f"дубликаты строк: {dup*100:.1f}% (это может искажать агрегаты).",
            base=0.8, surprise=(1.2 if dup >= 0.05 else 0.6)
        ))

    # main numeric
    if num_cols:
        main_num = sorted(num_cols, key=lambda c: (_relevance(qt, c), -df[c].isna().mean()), reverse=True)[0]
        s = pd.to_numeric(df[main_num], errors="coerce").dropna()
        if len(s) >= 5:
            p10, p50, p90 = np.percentile(s, [10, 50, 90])
            spread = (p90 / p50) if p50 not in (0, np.nan) else float("inf")
            sp = 1.6 if spread >= 3 else (0.9 if spread >= 2 else 0.0)
            cand.append(_cand(
                f"ключевая метрика «{main_num}»: медиана={p50:,.0f}, p10={p10:,.0f}, p90={p90:,.0f} (n={len(s)}).",
                base=1.3, surprise=sp, rel=_relevance(qt, main_num)
            ))
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out_rate = float(((s < lo) | (s > hi)).mean())
                if out_rate >= 0.05:
                    cand.append(_cand(
                        f"выбросы по «{main_num}» (iqr 1.5): {out_rate*100:.1f}%.",
                        base=0.9, surprise=(1.3 if out_rate >= 0.15 else 0.7), rel=_relevance(qt, main_num)
                    ))

    # main categorical (dominance / high cardinality)
    if cat_cols:
        main_cat = sorted(cat_cols, key=lambda c: (_relevance(qt, c), int(df[c].nunique(dropna=True))), reverse=True)[0]
        vc = df[main_cat].value_counts(dropna=True)
        if len(vc) > 0:
            top_val, top_cnt = str(vc.index[0]), int(vc.iloc[0])
            share = top_cnt / n
            if share >= 0.35:
                sp = 2.0 if share >= 0.7 else (1.2 if share >= 0.5 else 0.7)
                cand.append(_cand(
                    f"доминирующая категория в «{main_cat}»: «{top_val}» — {top_cnt} строк ({share*100:.1f}%).",
                    base=1.0, surprise=sp, rel=_relevance(qt, main_cat)
                ))
            nun = int(df[main_cat].nunique(dropna=True))
            if nun >= 50:
                cand.append(_cand(
                    f"высокое разнообразие в «{main_cat}»: {nun} уникальных значений (возможен top-n).",
                    base=0.8, surprise=0.8, rel=_relevance(qt, main_cat)
                ))

    cand = sorted(cand, key=lambda x: x["score"], reverse=True)

    facts, seen = [], set()
    for c in cand:
        t = c["text"]
        key = re.sub(r"\d+", "#", t.lower())
        if key in seen:
            continue
        seen.add(key)
        facts.append(t)
        if len(facts) == 3:
            break

    while len(facts) < 3:
        facts.append("данные получены; следующий шаг — выбор визуализации и построение графика.")

    return facts
