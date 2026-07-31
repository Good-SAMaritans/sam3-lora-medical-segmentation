"""
append one row per run to a CSV
"""

import csv
import os


def completed_runs(csv_path: str) -> set[str]:
    """Run names already in the CSV, so a sweep can resume where it stoppe"""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["run"] for row in csv.DictReader(f)}


def log_run(csv_path: str, row: dict) -> str:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return csv_path
