# Test 6: how much data each dataset gives to train and test, in windows and hours, against the hours in the paper.
import argparse
import json
from pathlib import Path

import numpy as np

from data import DATASETS, ROOT

# Section 4.1 of the paper, hours; MIMIC-AFib is not given
PAPER_HOURS = {"WESAD": 24, "DALIA": 35, "CAPNO": 5.6, "BIDMC": 7, "MIMIC-AFib": None}
PAPER_SUBJECTS = {"WESAD": 15, "DALIA": 15, "CAPNO": 42, "BIDMC": 53, "MIMIC-AFib": 35}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = {}
    totals = {"train": 0, "test": 0}
    for name in args.datasets:
        folder = Path(args.data_root) / name
        splits = json.load(open(folder / "splits.json"))
        row = {"subjects_train": len(splits["train"]), "subjects_test": len(splits["test"]),
               "paper_subjects": PAPER_SUBJECTS[name], "paper_hours": PAPER_HOURS[name]}
        for part in ["train", "test"]:
            windows = len(np.load(folder / f"ecg_{part}_4sec.npy", mmap_mode="r"))
            row[f"windows_{part}"] = windows
            row[f"hours_{part}"] = windows * 4 / 3600
            totals[part] += windows
        row["hours_total"] = row["hours_train"] + row["hours_test"]
        results[name] = row
        print(f"{name}: subjects {row['subjects_train']}+{row['subjects_test']} (paper {row['paper_subjects']}), "
              f"hours train {row['hours_train']:.1f} test {row['hours_test']:.1f} total {row['hours_total']:.1f} (paper {row['paper_hours']})", flush=True)

    results["total"] = {part: {"windows": n, "hours": n * 4 / 3600} for part, n in totals.items()}
    print(f"total train {totals['train']} windows = {totals['train'] * 4 / 3600:.1f} h, test {totals['test']} = {totals['test'] * 4 / 3600:.1f} h")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
