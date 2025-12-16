def check_db(con):
    tables = con.execute("SHOW TABLES").fetchall()
    if not tables:
        raise RuntimeError("Database has no tables")

    vacancies = con.execute("SELECT COUNT(*) FROM Vacancies").fetchone()[0]
    if vacancies == 0:
        raise RuntimeError("Vacancies table is empty")

    return {
        "tables": [t[0] for t in tables],
        "vacancies": vacancies,
    }
