"""splitting.py
Make and check a stratified train/validation/test split.

Two steps (test set first, then validation from the rest), 70/15/15 by default, and
seed = 42 + iteration: the same split as the benchmark models in the project explorer, so
your iteration-N split is the benchmark's iteration-N split.

Usage:

from splitting import stratified_split, validate_split
splits = stratified_split(df, target_column="binary_label", id_column="Id", seed=43)  # iteration 1
summary = validate_split(splits, df, target_column="binary_label", id_column="Id")
train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from loading import check_required_columns

PARTITIONS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
BASE_SEED = 42


class SplitValidationError(ValueError):
    """Raised when the split can't be made, or fails a check in validate_split()."""


# --------------------------------------------------------------
# Split
# --------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    target_column: str,
    id_column: str,
    fractions: dict = DEFAULT_FRACTIONS,
    seed: int = BASE_SEED,
) -> dict:
    """Split `df` into {"train", "val", "test"} DataFrames that keep the class balance of `target_column`.

    Rows are sorted by `id_column` first, so the split depends only on the data and the
    seed (not on row order in the CSV). The test set is split off first, then the
    validation set is taken from what is left.
    """
    ordered = df.sort_values(id_column, kind="stable").reset_index(drop=True)

    try:
        train_val, test = train_test_split(
            ordered,
            test_size=fractions["test"],
            stratify=ordered[target_column],
            random_state=seed,
        )
        train, val = train_test_split(
            train_val,
            test_size=fractions["val"] / (fractions["train"] + fractions["val"]),
            stratify=train_val[target_column],
            random_state=seed,
        )
    except ValueError as err:
        raise SplitValidationError(
            f"scikit-learn could not make a stratified split: {err}\n"
            f"Check the class counts of {target_column}, or use larger val/test fractions."
        ) from err

    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        print(f"{name:>5}: {len(part):,} rows ({len(part) / len(ordered):.1%})")
    return {name: part.reset_index(drop=True) for name, part in splits.items()}


# --------------------------------------------------------------
# Validation
# --------------------------------------------------------------
def validate_split(
    splits: dict,
    full_df: pd.DataFrame,
    target_column: str,
    id_column: str,
    fractions: dict = DEFAULT_FRACTIONS,
    tolerance: float = 0.02,
) -> pd.DataFrame:
    """Check a split and return a summary table (rows and class balance per partition).

    Raises SplitValidationError listing every failed check:
      * train / val / test all exist and are non-empty
      * no ID appears twice, or in more than one partition
      * together the partitions contain every row of `full_df` exactly once
      * every class appears in every partition
      * partition sizes and class proportions are within `tolerance` of the
        requested fractions and the overall class proportions
    """
    if not isinstance(splits, dict) or set(splits) != set(PARTITIONS):
        found = list(splits) if isinstance(splits, dict) else type(splits).__name__
        raise SplitValidationError(f"Expected a dict with keys {list(PARTITIONS)}, got {found}")
    check_required_columns(full_df, [id_column, target_column], "patient table")
    for name in PARTITIONS:
        check_required_columns(splits[name], [id_column, target_column], f"{name} split")

    problems = []
    n_all = len(full_df)

    id_sets = {name: set(splits[name][id_column]) for name in PARTITIONS}
    for name in PARTITIONS:
        if splits[name].empty:
            problems.append(f"The {name} split is empty.")
        n_repeated = len(splits[name]) - len(id_sets[name])
        if n_repeated:
            problems.append(f"{name} repeats {n_repeated:,} {id_column} value(s).")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = id_sets[a] & id_sets[b]
        if overlap:
            examples = sorted(map(str, overlap))[:3]
            problems.append(
                f"{len(overlap):,} {id_column} value(s) are in both {a} and {b} (data leakage), e.g. {examples}"
            )

    input_ids = set(full_df[id_column])
    split_ids = set().union(*id_sets.values())
    n_split_rows = sum(len(splits[name]) for name in PARTITIONS)
    if split_ids != input_ids or n_split_rows != n_all:
        problems.append(
            f"The partitions do not cover the input exactly once: input has {n_all:,} rows, "
            f"partitions have {n_split_rows:,}; {len(input_ids - split_ids):,} input IDs are in no "
            f"partition and {len(split_ids - input_ids):,} partition IDs are not in the input."
        )

    # .tolist()/.to_dict() give plain Python values, so messages show 1 rather than np.int64(1)
    classes = sorted(full_df[target_column].unique().tolist(), key=str)
    overall_share = full_df[target_column].value_counts(normalize=True).to_dict()
    rows = []
    for name in PARTITIONS + ("all",):
        part = full_df if name == "all" else splits[name]
        n = len(part)
        counts = part[target_column].value_counts().to_dict()
        row = {"partition": name, "rows": n, "pct_of_all_rows": round(100 * n / n_all, 1)}
        for cls in classes:
            row[f"class_{cls}_rows"] = int(counts.get(cls, 0))
            row[f"class_{cls}_pct"] = round(100 * counts.get(cls, 0) / n, 1) if n else 0.0
        rows.append(row)

        if name == "all" or n == 0:
            continue
        if abs(n / n_all - fractions[name]) > tolerance + 1 / n_all:
            problems.append(f"{name} has {n / n_all:.1%} of the rows, but {fractions[name]:.1%} was requested.")
        allowed = tolerance + 1 / n
        for cls in classes:
            share = counts.get(cls, 0) / n
            if counts.get(cls, 0) == 0:
                problems.append(f"Class {cls!r} is missing from the {name} split.")
            elif abs(share - overall_share[cls]) > allowed:
                problems.append(
                    f"Class {cls!r} is {share:.1%} of {name} but {overall_share[cls]:.1%} overall "
                    f"(allowed difference {allowed:.1%})."
                )

    if problems:
        raise SplitValidationError("Split validation failed:\n  - " + "\n  - ".join(problems))
    print(
        "All split checks passed: partitions are disjoint, cover every row once, "
        "and keep the class balance."
    )
    return pd.DataFrame(rows).set_index("partition")
