import logging
from data.db import get_db_con

log = logging.getLogger(__name__)


def init_db(data_path: str = "data/vacancies.json"):
    log.info("📦 Loading data into in-memory DuckDB")

    con = get_db_con(data_path)

    tables = con.execute("SHOW TABLES").fetchall()
    log.info("📊 Tables loaded: %s", [t[0] for t in tables])

    count = con.execute("SELECT COUNT(*) FROM Vacancies").fetchone()[0]
    log.info("📈 Vacancies loaded: %s", count)

    return con
