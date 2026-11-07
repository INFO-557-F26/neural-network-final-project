"""loading.py
List and load the project's CSV files from a folder (on Cacao JupyterHub: /home/shared).

Usage:

from loading import list_csv_files, load_csv_files
list_csv_files("/home/shared")                         # which CSV files are there, and how big
tables = load_csv_files("/home/shared", ["patients"])  # {"patients": DataFrame}
"""

from pathlib import Path

import pandas as pd


def _csv_paths(folder) -> dict:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return {path.stem: path for path in sorted(folder.glob("*.csv"))}


def list_csv_files(folder) -> pd.DataFrame:
    """One row per CSV file directly inside `folder`: table name, file name and size."""
    rows = [
        {"table": name, "file": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
        for name, path in _csv_paths(folder).items()
    ]
    return pd.DataFrame(rows, columns=["table", "file", "size_mb"])


def load_csv_files(folder, tables=None) -> dict:
    """Load CSV files from `folder` into {table name: DataFrame}.

    tables: list of table names (file names without .csv), or None to load every CSV.
    """
    paths = _csv_paths(folder)
    if isinstance(tables, str):
        tables = [tables]
    selected = list(paths) if tables is None else list(tables)
    missing = [name for name in selected if name not in paths]
    if missing:
        raise FileNotFoundError(f"Table(s) {missing} not found in {folder}. Available tables: {list(paths)}")

    loaded = {}
    for name in selected:
        df = pd.read_csv(paths[name], low_memory=False)
        print(f"Loaded {paths[name].name}: {len(df):,} rows x {df.shape[1]} columns")
        loaded[name] = df
    return loaded


def check_required_columns(df: pd.DataFrame, columns, table_name: str = "table") -> None:
    """Raise a KeyError that lists what is missing (and near matches) if any column is absent."""
    missing = [col for col in columns if col not in df.columns]
    if not missing:
        return
    lower_to_actual = {str(col).lower(): col for col in df.columns}
    hints = [
        f"'{col}' -> did you mean '{lower_to_actual[str(col).lower()]}'?"
        for col in missing
        if str(col).lower() in lower_to_actual
    ]
    message = (
        f"{table_name} is missing required column(s): {missing}\n"
        f"Available columns: {list(df.columns)}"
    )
    if hints:
        message += "\nColumn names are case-sensitive: " + "; ".join(hints)
    raise KeyError(message)
