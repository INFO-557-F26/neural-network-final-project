"""outputs.py
Name, place, write and check the files you submit for the final project.

Every run (one iteration) writes these files inside one analysis folder, e.g.
analysis/my-best-model-analysis/:

    results/metrics/<model>-metrics-iter<N>.csv            one row: iteration, seed, metrics, compute cost
    results/figures/<model>-confusion-matrix-iter<N>.png   confusion matrix on the test split
    results/figures/<model>-roc-curve-iter<N>.png          ROC curve on the test split
    logs/<model>-iter<N>.out                               everything the run printed
    logs/resource-usage.csv                                one row per completed run (ResourceLogger)

<model> is "my-model" for your project ("logistic-regression" in example-for-guide/).
combine_iteration_metrics() writes results/metrics/<model>-metrics-all-iterations.csv.
ResourceLogger also adds a row for every run to analysis/overall-resource-usage-summary.csv;
check_budget() summarizes it.

Usage:

from outputs import output_paths, capture_output, save_test_results
paths = output_paths("analysis/my-best-model-analysis", "my-model", iteration=1)
with capture_output(paths["log"]):                      # copies everything printed into the .out file
    with ResourceLogger(tag="my-model-iter1", log_path=paths["resource_usage"]) as usage:
        test_prob = ...                                 # train your model, predict the test split
    save_test_results(paths, y_test, test_prob, iteration=1, seed=43, usage=usage)
"""

import re
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd

from evaluation import METRIC_NAMES, compute_metrics, plot_confusion_matrix, plot_roc_curve
from logger import BUDGET_HR, USAGE_SUMMARY_PATH, budget_status, display_path, read_usage_summary

METRICS_COLUMNS = ["iteration", "seed", *METRIC_NAMES, "elapsed_hr", "gpu_hrs", "cpu_pct"]
USAGE_COLUMNS = ["elapsed_hr", "gpu_hrs", "cpu_pct"]
CODE_SUFFIXES = {".py", ".ipynb"}  # your scripts may sit next to their outputs
# Fixed decimals in the CSV, so short runs show 0.000002 rather than 2e-06
COLUMN_FORMATS = {**{name: "{:.4f}" for name in METRIC_NAMES},
                  "elapsed_hr": "{:.6f}", "gpu_hrs": "{:.6f}", "cpu_pct": "{:.1f}"}


def _write_metrics_csv(table: pd.DataFrame, path) -> None:
    formatted = table.copy()
    for name, fmt in COLUMN_FORMATS.items():
        formatted[name] = formatted[name].map(fmt.format)
    formatted.to_csv(path, index=False)


# --------------------------------------------------------------
# File names
# --------------------------------------------------------------
def output_paths(output_dir, model_name: str = "my-model", iteration: int = 1) -> dict:
    """Return the path of every output file for one iteration, and create the folders.

    Keys: metrics, confusion_matrix, roc_curve, log, resource_usage, all_iterations.
    """
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(model_name)):
        raise ValueError(
            f"model_name {model_name!r} must use only lowercase letters, digits and '-', e.g. 'my-model'"
        )
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError(f"iteration must be a whole number 1, 2, 3, ... (got {iteration!r})")

    output_dir = Path(output_dir)
    metrics_dir = output_dir / "results" / "metrics"
    figures_dir = output_dir / "results" / "figures"
    logs_dir = output_dir / "logs"
    for folder in (metrics_dir, figures_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    run = f"iter{iteration}"
    return {
        "metrics": metrics_dir / f"{model_name}-metrics-{run}.csv",
        "confusion_matrix": figures_dir / f"{model_name}-confusion-matrix-{run}.png",
        "roc_curve": figures_dir / f"{model_name}-roc-curve-{run}.png",
        "log": logs_dir / f"{model_name}-{run}.out",
        "resource_usage": logs_dir / "resource-usage.csv",
        "all_iterations": metrics_dir / f"{model_name}-metrics-all-iterations.csv",
    }


# --------------------------------------------------------------
# .out log file
# --------------------------------------------------------------
class _Tee:
    """Write text to the screen and to a log file at the same time."""

    def __init__(self, screen, log_file):
        self.screen, self.log_file = screen, log_file

    def write(self, text):
        self.screen.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.screen.flush()
        self.log_file.flush()

    def __getattr__(self, name):  # e.g. .encoding or .isatty(), used by progress bars
        return getattr(self.screen, name)


@contextmanager
def capture_output(log_path):
    """Show printed output as usual and also save it (and any error) to log_path.

    The file is rewritten each time, so it always holds the latest run of that iteration.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
            print(f"Run started {datetime.now():%Y-%m-%d %H:%M:%S}; output saved to {display_path(log_path)}")
            try:
                yield log_path
            except BaseException:
                log_file.write(traceback.format_exc())  # the screen shows the error anyway
                raise
            finally:
                print(f"Run finished {datetime.now():%Y-%m-%d %H:%M:%S}")


# --------------------------------------------------------------
# Metrics file and figures
# --------------------------------------------------------------
def save_metrics(path, iteration: int, seed: int, metrics: dict, usage: dict) -> pd.DataFrame:
    """Write the one-row metrics file: iteration, seed, Accuracy ... F1_1, elapsed_hr, gpu_hrs, cpu_pct."""
    missing = [name for name in METRIC_NAMES if name not in metrics]
    if missing:
        raise KeyError(f"metrics is missing {missing}. Use evaluation.compute_metrics() to make it.")
    if any(name not in usage for name in USAGE_COLUMNS):
        raise KeyError(
            "usage has no resource numbers yet. Call this after the `with ResourceLogger(...) as usage:` "
            "block has finished (not inside it)."
        )
    row = {"iteration": int(iteration), "seed": int(seed)}
    row.update({name: round(float(metrics[name]), 4) for name in METRIC_NAMES})
    row.update({name: usage[name] for name in USAGE_COLUMNS})

    table = pd.DataFrame([row], columns=METRICS_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_metrics_csv(table, path)
    print(f"Saved metrics to {display_path(path)}")
    return table


def save_test_results(paths: dict, y_true, y_prob, iteration: int, seed: int, usage: dict,
                      threshold: float = 0.5) -> pd.DataFrame:
    """Compute the test metrics, then save the metrics CSV, confusion matrix and ROC curve.

    paths: the dict from output_paths(). y_prob: predicted probability of class 1 for each
    test row. usage: the dict from `with ResourceLogger(...) as usage`. Returns the metrics row.
    """
    run_name = Path(paths["log"]).stem  # e.g. my-model-iter1
    metrics = compute_metrics(y_true, y_prob, threshold)
    table = save_metrics(paths["metrics"], iteration, seed, metrics, usage)
    plot_confusion_matrix(y_true, y_prob, paths["confusion_matrix"], threshold, name=run_name)
    plot_roc_curve(y_true, y_prob, paths["roc_curve"], name=run_name)
    print(f"\nTest metrics for {run_name}:")
    print(table.to_string(index=False, formatters={k: v.format for k, v in COLUMN_FORMATS.items()}))
    return table


# --------------------------------------------------------------
# All iterations
# --------------------------------------------------------------
def combine_iteration_metrics(output_dir, model_name: str = "my-model", save: bool = True) -> pd.DataFrame:
    """Stack every results/metrics/<model_name>-metrics-iter<N>.csv in output_dir into one table.

    Prints the mean and standard deviation of each metric and, if save is True, writes
    results/metrics/<model_name>-metrics-all-iterations.csv.
    """
    metrics_dir = Path(output_dir) / "results" / "metrics"
    pattern = re.compile(rf"{re.escape(model_name)}-metrics-iter(\d+)\.csv")
    found = sorted(
        (int(match.group(1)), path)
        for path in metrics_dir.glob("*.csv")
        if (match := pattern.fullmatch(path.name))
    )
    if not found:
        raise FileNotFoundError(f"No {model_name}-metrics-iter<N>.csv files found in {display_path(metrics_dir)}")

    tables = []
    for _, path in found:
        table = pd.read_csv(path)
        if list(table.columns) != METRICS_COLUMNS or len(table) != 1:
            raise ValueError(
                f"{path.name} should have one row with columns {METRICS_COLUMNS}; "
                "re-create it with save_test_results()."
            )
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)

    print(f"{len(combined)} iteration(s) found: {[number for number, _ in found]}")
    print(combined[METRIC_NAMES].agg(["mean", "std"]).round(4).to_string())
    if save:
        save_path = metrics_dir / f"{model_name}-metrics-all-iterations.csv"
        _write_metrics_csv(combined, save_path)
        print(f"Saved all iterations to {display_path(save_path)}")
    return combined


def check_outputs(output_dir, model_names="my-model", iterations=range(1, 6),
                  all_iterations_file: bool = True) -> pd.DataFrame:
    """List every expected output file and whether it exists. Run it before you push.

    model_names: one name, or a list for task variants (e.g. ["my-model-size100", "my-model-size1000"]).
    Also flags files that do not follow the naming convention, and runs missing from resource-usage.csv.
    """
    output_dir = Path(output_dir)
    model_names = [model_names] if isinstance(model_names, str) else list(model_names)
    iterations = [int(iteration) for iteration in iterations]
    if not model_names or not iterations:
        raise ValueError("Pass at least one model name and one iteration, e.g. 'my-model', range(1, 6)")

    expected = []
    for name in model_names:
        for iteration in iterations:
            paths = output_paths(output_dir, name, iteration)
            expected += [paths["metrics"], paths["confusion_matrix"], paths["roc_curve"], paths["log"]]
        if all_iterations_file:
            expected.append(paths["all_iterations"])
    expected.append(paths["resource_usage"])

    rows = [{"file": p.relative_to(output_dir).as_posix(), "status": "ok" if p.exists() else "MISSING"}
            for p in expected]
    expected_set = set(expected)
    prefixes = tuple(f"{name}-" for name in model_names)
    for path in sorted(output_dir.rglob("*")):
        if (path.is_file() and path.name != ".gitkeep" and path.suffix not in CODE_SUFFIXES
                and path not in expected_set):
            status = ("extra: push only if a task asks for it" if path.name.startswith(prefixes)
                      else f"wrong name: must start with {model_names[0]}- (or remove it)")
            rows.append({"file": path.relative_to(output_dir).as_posix(), "status": status})

    usage_path = paths["resource_usage"]
    if usage_path.exists():
        logged_tags = set(pd.read_csv(usage_path)["tag"].astype(str))
        for name in model_names:
            for iteration in iterations:
                if f"{name}-iter{iteration}" not in logged_tags:
                    rows.append({"file": f"logs/resource-usage.csv (tag {name}-iter{iteration})",
                                 "status": "MISSING row"})

    report = pd.DataFrame(rows, columns=["file", "status"])
    problems = report[report["status"] != "ok"]
    if problems.empty:
        print(f"All {len(report)} expected files are in {display_path(output_dir)}.")
    else:
        print(f"{len(problems)} problem(s) in {display_path(output_dir)}: fix them before you push.")
    return report


# --------------------------------------------------------------
# Compute budget
# --------------------------------------------------------------
def check_budget(summary_path=USAGE_SUMMARY_PATH) -> pd.DataFrame:
    """Total logged run time per analysis folder and for the whole project, against the budget.

    Reads analysis/overall-resource-usage-summary.csv (every run, written by ResourceLogger),
    prints the project total and its budget status, and returns one row per folder.
    """
    summary_path = Path(summary_path)
    rows = read_usage_summary(summary_path)
    columns = ["analysis_folder", "runs", "failed_or_interrupted", "time_needed_hr"]
    if not rows:
        print(f"No runs logged yet in {display_path(summary_path)}. Budget {BUDGET_HR:g} h: {budget_status(0)}")
        return pd.DataFrame(columns=columns)
    runs = pd.DataFrame(rows)
    hours = runs["time_needed_hr"].astype(float)
    runs["time_needed_hr"] = hours

    # The file checks itself: run numbers count up from 1 and the running total adds up
    consistent = (runs["run"].astype(int).tolist() == list(range(1, len(runs) + 1))
                  and (runs["total_time_hr"].astype(float) - hours.cumsum()).abs().max() < 1e-5)
    if not consistent:
        print(f"WARNING: the run numbers or running totals in {display_path(summary_path)} do not add up. "
              "Rows were edited or deleted. Tell your instructor.")

    per_folder = (
        runs.assign(failed_or_interrupted=runs["status"] != "completed")
        .groupby("analysis_folder", sort=True)
        .agg(runs=("run", "size"), failed_or_interrupted=("failed_or_interrupted", "sum"),
             time_needed_hr=("time_needed_hr", "sum"))
        .reset_index()
    )
    total_hr = float(hours.sum())
    print(f"{len(runs)} runs logged, {total_hr:.3f} h in total, budget {BUDGET_HR:g} h: {budget_status(total_hr)}")
    return per_folder[columns]
