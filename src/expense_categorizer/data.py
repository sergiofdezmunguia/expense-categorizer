"""Lectura de solo lectura de la base local de Personal OS. Los datos nunca salen del PC ni
entran en el repositorio: la evaluación real escribe en `results/real/` (ignorado por git)."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB = (
    Path(os.environ.get("POS_DATA_DIR", "~/.local/share/personal-os")).expanduser()
    / "pos.db"
)


def load_personal_os(db: Path = DEFAULT_DB) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(
            "SELECT t.booking_date, t.description, t.amount_cents, a.kind AS account_kind,"
            " t.category, t.category_source"
            " FROM fin_transactions t JOIN fin_accounts a ON a.id = t.account_id"
            " ORDER BY t.booking_date, t.rowid",
            con,
        )
    finally:
        con.close()
