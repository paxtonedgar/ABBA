from .duckdb import Storage, StorageValidationError

__all__ = ["Storage", "StorageValidationError", "create_storage"]


def create_storage(db_path: str = ":memory:", pg_dsn: str | None = None) -> Storage:
    """Factory: returns DualStorage if Postgres DSN is available, else plain DuckDB.

    Checks pg_dsn arg first, then SUPABASE_DB_URL env var.
    """
    import os
    dsn = pg_dsn or os.environ.get("SUPABASE_DB_URL", "")
    if dsn:
        from .dual import DualStorage
        return DualStorage(db_path=db_path, pg_dsn=dsn)
    return Storage(db_path=db_path)
