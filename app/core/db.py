# app/core/db.py
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

DEFAULT_DB_PATH = Path("data/processed/fraud.db")
DEFAULT_CSV_PATH = Path("data/raw/fraudTrain.csv")
TABLE_NAME = "transactions"


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Pragmas for decent performance on local file DB
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Initialize transactions table if not exists.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            trans_date_trans_time TEXT,
            cc_num TEXT,
            merchant TEXT,
            category TEXT,
            amt REAL,
            first TEXT,
            last TEXT,
            gender TEXT,
            street TEXT,
            city TEXT,
            state TEXT,
            zip TEXT,
            lat REAL,
            long REAL,
            city_pop INTEGER,
            job TEXT,
            dob TEXT,
            trans_num TEXT,
            unix_time INTEGER,
            merch_lat REAL,
            merch_long REAL,
            is_fraud INTEGER
        )
        """
    )
    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """
    Create indexes used for analytical queries.
    """
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_time ON {TABLE_NAME}(trans_date_trans_time)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_merchant ON {TABLE_NAME}(merchant)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_category ON {TABLE_NAME}(category)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_is_fraud ON {TABLE_NAME}(is_fraud)")
    conn.commit()


def _normalize_datetime_series(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def load_csv_to_sqlite(
    conn: sqlite3.Connection,
    csv_path: Path = DEFAULT_CSV_PATH,
    chunksize: int = 50_000,
    log_every_n_chunks: int = 5,
) -> None:
    """
    Load CSV into SQLite in chunks with basic normalization.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    t0 = time.perf_counter()
    logger.info(
        "Loading CSV to SQLite | csv=%s | chunksize=%d | table=%s",
        csv_path,
        chunksize,
        TABLE_NAME,
    )

    reader = pd.read_csv(csv_path, chunksize=chunksize)

    total_rows = 0
    chunk_i = 0

    for chunk in reader:
        chunk_i += 1

        # Drop junk columns
        for junk in ["Unnamed: 0", "index", "__index_level_0__"]:
            if junk in chunk.columns:
                chunk = chunk.drop(columns=[junk])

        # Normalize datetime
        if "trans_date_trans_time" in chunk.columns:
            chunk["trans_date_trans_time"] = _normalize_datetime_series(
                chunk["trans_date_trans_time"]
            )

        # Normalize DOB
        if "dob" in chunk.columns:
            chunk["dob"] = (
                pd.to_datetime(chunk["dob"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )

        # Ensure fraud flag is int 0/1
        if "is_fraud" in chunk.columns:
            chunk["is_fraud"] = (
                pd.to_numeric(chunk["is_fraud"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

        chunk.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        total_rows += len(chunk)
        if log_every_n_chunks > 0 and chunk_i % log_every_n_chunks == 0:
            logger.info(
                "Inserted progress | chunks=%d | rows=%d",
                chunk_i,
                total_rows,
            )

    conn.commit()
    logger.info(
        "CSV load done | chunks=%d | rows=%d | elapsed=%.2fs",
        chunk_i,
        total_rows,
        time.perf_counter() - t0,
    )


def bootstrap_db(
    db_path: Path = DEFAULT_DB_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    force_reload: bool = False,
) -> Path:
    """
    Ensure SQLite DB exists and is populated.
    - If empty -> load CSV + create indexes
    - If force_reload -> clear table and reload
    """
    conn = get_connection(db_path)
    try:
        t0 = time.perf_counter()
        logger.info(
            "Bootstrap DB start | db=%s | csv=%s | force_reload=%s",
            db_path,
            csv_path,
            force_reload,
        )

        init_schema(conn)

        if force_reload:
            logger.warning(
                "Force reload enabled: clearing table | table=%s",
                TABLE_NAME,
            )
            conn.execute(f"DELETE FROM {TABLE_NAME}")
            conn.commit()

        row = conn.execute(
            f"SELECT COUNT(1) AS n FROM {TABLE_NAME}"
        ).fetchone()
        n = int(row["n"]) if row else 0
        logger.info(
            "Existing rows | table=%s | rows=%d",
            TABLE_NAME,
            n,
        )

        if n == 0:
            load_csv_to_sqlite(conn, csv_path=csv_path)
            create_indexes(conn)
        else:
            create_indexes(conn)

        logger.info(
            "Bootstrap DB done | elapsed=%.2fs",
            time.perf_counter() - t0,
        )

    finally:
        conn.close()

    return db_path


if __name__ == "__main__":
    bootstrap_db()
    print(f"SQLite ready: {DEFAULT_DB_PATH}")
