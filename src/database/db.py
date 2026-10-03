from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.config import DB_URL
from src.database.models import Base


def make_engine(url: str = DB_URL):
    kwargs = {"connect_args": {"check_same_thread": False, "timeout": 30}} if url.startswith("sqlite") else {}
    return create_engine(url, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(bind=None):
    bind = bind or engine
    Base.metadata.create_all(bind=bind)
    _add_missing_columns(bind)


def _add_missing_columns(bind):
    """Tiny migration: when a newer version adds columns, add them to an existing DB instead of failing.
    Existing rows are kept and get the column default."""
    insp = inspect(bind)
    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = col.type.compile(dialect=bind.dialect)
                default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                sql = f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'
                if isinstance(default, str):
                    sql += " DEFAULT '" + default.replace("'", "''") + "'"
                elif isinstance(default, (int, float)):
                    sql += f" DEFAULT {default}"
                conn.execute(text(sql))


def check_connection(bind=None) -> bool:
    with (bind or engine).connect() as conn:
        return conn.execute(text("SELECT 1")).scalar() == 1


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
