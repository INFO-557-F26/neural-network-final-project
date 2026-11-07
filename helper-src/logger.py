"""logger.py
Log wall-clock runtime, CPU/GPU usage, and disk usage of a run to CSV, and track the
project's compute budget.

Usage:

from logger import ResourceLogger
with ResourceLogger(tag="my-model-iter1", log_path=paths["resource_usage"]) as usage:
    train()
print(usage["elapsed_hr"], usage["gpu_hrs"], usage["cpu_pct"])

Every run writes to two files:
  * log_path, normally analysis/<task>-analysis/logs/resource-usage.csv: one row per
    COMPLETED run (outputs.output_paths() gives the path);
  * analysis/overall-resource-usage-summary.csv: one row for EVERY run of the whole
    project (completed, failed or interrupted) with its start time, end time and time
    needed, the running total, and the budget status. Every run counts toward the budget.
The same summary row is also sent to the course resource sheet, with the JupyterHub username
('local' when the run is not on JupyterHub). Sending is silent and never stops a run.
After the block, `usage` holds the measured numbers, so they can go into your metrics file.

resource-usage.csv columns: tag, elapsed_hr, gpu_hrs, cpu_pct, disk_used_gb. torch is optional
(without it, gpu_hrs is 0.0). Times are written with 6 decimals so short runs do not show as 0.000.
"""

import csv
import json
import os
import shutil
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import psutil

try:
    import torch
except ImportError:  # torch is only used to check whether a GPU is available
    torch = None

HEADER = ["tag", "elapsed_hr", "gpu_hrs", "cpu_pct", "disk_used_gb"]

REPO_ROOT = Path(__file__).resolve().parents[1]

# Overall summary: one row per run of the whole project. The course sheet has the same
# columns, plus received_at (the time the sheet got the row).
USAGE_SUMMARY_PATH = REPO_ROOT / "analysis" / "overall-resource-usage-summary.csv"
SUMMARY_HEADER = [
    "jupyterhub_username", "run", "analysis_folder", "tag", "status", "start_time", "end_time",
    "time_needed", "time_needed_hr", "gpu_hrs", "cpu_pct", "total_time_hr", "budget_status",
]
LOCAL_USERNAME = "local"  # jupyterhub_username when the run is not on JupyterHub

# Course resource sheet (instructor-owned; the web app appends rows, students can only view)
SHEET_URL = "https://script.google.com/macros/s/AKfycbwhaTz1rsBPlX_QGuU7VBYFvA7BJsVSTYtwzQCywbUJjNudj6MpURVkL0Tc1VehRg82Rw/exec"
SHEET_TIMEOUT_S = 15

# Compute budget for the whole project, in hours of logged run time (every run counts)
BUDGET_HR = 4.0  # up to 4 h: full credit
FLAG_HR = 5.0    # 4-5 h: full credit, flagged
MAX_HR = 8.0     # 5-8 h: budget points lost in proportion; over 8 h: all budget points lost


def budget_status(total_hr: float) -> str:
    """Budget tier for a project total, in hours."""
    if total_hr <= BUDGET_HR:
        return "full credit"
    if total_hr <= FLAG_HR:
        return "full credit, flagged"
    if total_hr < MAX_HR:
        kept = (MAX_HR - total_hr) / (MAX_HR - FLAG_HR)
        return f"partial credit: {kept:.0%} of budget points"
    return "no budget points"


def read_usage_summary(summary_path=USAGE_SUMMARY_PATH) -> list:
    """Rows of the overall summary file as dicts ([] if it does not exist yet), or raise if it was changed."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return []
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    try:
        if reader.fieldnames != SUMMARY_HEADER:
            raise ValueError(f"its columns are {reader.fieldnames}")
        for row in rows:
            float(row["time_needed_hr"])
    except (ValueError, TypeError) as err:
        raise ValueError(
            f"{summary_path} is not in the format the resource logger writes ({err}).\n"
            "It is written by the resource logger only: do not edit it or save it from Excel. "
            "Tell your instructor before running again."
        ) from None
    return rows


def display_path(path: Path) -> str:
    """Path relative to the current folder when possible, with forward slashes."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def jupyterhub_username() -> str:
    """JupyterHub user name of this run (set by JupyterHub), or 'local' when not on JupyterHub."""
    return os.environ.get("JUPYTERHUB_USER", "").strip() or LOCAL_USERNAME


def _send_to_sheet(row: dict) -> bool:
    """Send one summary row to the course sheet, silently. True if the sheet saved it; never raises."""
    try:
        request = urllib.request.Request(
            SHEET_URL, data=json.dumps(row).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=SHEET_TIMEOUT_S) as response:
            return bool(json.loads(response.read().decode("utf-8") or "{}").get("ok"))
    except Exception:  # offline, timeout, sheet down: the local file still has the run
        return False


def _append_summary_row(summary_path: Path, values: dict) -> dict:
    rows = read_usage_summary(summary_path)
    total_hr = sum(float(row["time_needed_hr"]) for row in rows) + values["time_needed_hr"]
    row = {
        **values,
        "run": len(rows) + 1,
        "time_needed_hr": f"{values['time_needed_hr']:.6f}",
        "gpu_hrs": f"{values['gpu_hrs']:.6f}",
        "cpu_pct": f"{values['cpu_pct']:.1f}",
        "total_time_hr": f"{total_hr:.6f}",
        "budget_status": budget_status(total_hr),
    }
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_HEADER, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


@contextmanager
def ResourceLogger(tag: str, log_path, summary_path=USAGE_SUMMARY_PATH):
    if not str(tag).strip():
        raise ValueError("ResourceLogger needs a non-empty tag, e.g. tag='my-model-iter1'")
    log_path, summary_path = Path(log_path), Path(summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    read_usage_summary(summary_path)  # stop before training if the summary file was changed
    username = jupyterhub_username()
    folder = log_path.parent.parent.name if log_path.parent.name == "logs" else log_path.parent.name
    usage = {}  # filled in after the block finishes

    psutil.cpu_percent(interval=None)  # start the CPU measurement window here
    started = datetime.now()
    start = time.time()
    status = "completed"
    try:
        yield usage  # the code inside the with-block runs here
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        # Every run is recorded, including failed and interrupted ones
        end = time.time()
        ended = datetime.now()

        # wall time (hours)
        elapsed_hr = (end - start) / 3600

        # CPU percent: average system-wide utilisation since the block started
        cpu_pct = psutil.cpu_percent(interval=None)

        # GPU hours (simple estimate: 1 GPU * elapsed_hr if CUDA present)
        gpu_hrs = elapsed_hr if torch is not None and torch.cuda.is_available() else 0.0

        # Disk used (GB) after run
        _, used, _ = shutil.disk_usage("/")
        used_gb = used / 1e9

        # Same rounding as the CSV, so the metrics file and the log agree
        usage.update(
            tag=tag,
            elapsed_hr=round(elapsed_hr, 6),
            gpu_hrs=round(gpu_hrs, 6),
            cpu_pct=round(cpu_pct, 1),
            disk_used_gb=round(used_gb, 1),
        )

        # Append to the folder's CSV (completed runs only)
        if status == "completed":
            row = [tag, f"{elapsed_hr:.6f}", f"{gpu_hrs:.6f}", f"{cpu_pct:.1f}", f"{used_gb:.1f}"]
            write_header = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                if write_header:
                    writer.writerow(HEADER)
                writer.writerow(row)

        # Append to the overall summary (every run)
        summary = _append_summary_row(summary_path, {
            "jupyterhub_username": username,
            "analysis_folder": folder,
            "tag": tag,
            "status": status,
            "start_time": started.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": ended.strftime("%Y-%m-%d %H:%M:%S"),
            "time_needed": str(timedelta(seconds=round(end - start))),
            "time_needed_hr": round(elapsed_hr, 6),
            "gpu_hrs": round(gpu_hrs, 6),
            "cpu_pct": round(cpu_pct, 1),
        })
        usage.update(run=summary["run"], total_time_hr=float(summary["total_time_hr"]))

        target = display_path(log_path) if status == "completed" else f"not added to {log_path.name}"
        print(
            f"[ResourceLogger] {tag} ({status}): {end - start:.1f} s, gpu_hrs={gpu_hrs:.6f}, "
            f"cpu_pct={cpu_pct:.1f}, disk_used_gb={used_gb:.1f} -> {target}"
        )
        print(
            f"[ResourceLogger] run {summary['run']}: project total {float(summary['total_time_hr']):.3f} h "
            f"of the {BUDGET_HR:g} h budget ({summary['budget_status']}) -> {display_path(summary_path)}"
        )
        if float(summary["total_time_hr"]) > BUDGET_HR:
            print(f"[ResourceLogger] WARNING: over the {BUDGET_HR:g} h compute budget. Every further run adds to the total.")

        # Same row to the course sheet, silently; a failure here never stops or fails the run
        _send_to_sheet(summary)
