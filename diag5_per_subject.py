# Test 5: heart-rate error per subject, for ours and the authors' weights, on our test and some of our train subjects.
# Windows are rebuilt per subject with preprocess.py, so the result does not depend on the order inside the .npy files.
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from diag_common import add_model_args, device_name, generate, hr_stats, load_models, rmse
from preprocess import LOADERS, process_subject


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--datasets", nargs="*", default=["WESAD", "DALIA"])
    parser.add_argument("--train-subjects", type=int, default=3, help="how many train subjects to add for comparison")
    parser.add_argument("--max-windows", type=int, default=300, help="8 s windows per subject, evenly spaced")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = device_name()
    models = load_models(args, device)
    root = Path(args.data_root)
    results = {}
    for name in args.datasets:
        splits = json.load(open(root / name / "splits.json"))
        listing = dict(LOADERS[name][0](root))
        chosen = [(s, "test") for s in splits["test"]] + [(s, "train") for s in splits["train"][:args.train_subjects]]
        total_test = 0
        for subject, part in chosen:
            out = process_subject((name, subject, listing[subject]))
            if out["error"] or 8 not in out["windows"]:
                print(f"{name} {subject}: skipped, {out['error']}", flush=True)
                continue
            windows = out["windows"][8]  # (windows, [ecg, ppg, roi], 1024)
            if part == "test":
                total_test += len(windows)
            keep = np.linspace(0, len(windows) - 1, min(args.max_windows, len(windows))).astype(int)
            real = windows[keep, 0].astype(np.float64)
            ppg = torch.from_numpy(windows[keep, 1]).float().unsqueeze(1)
            for model_name, model in models.items():
                fake = generate(model, ppg, device, args.seed)
                row = hr_stats(real, fake)
                row["rmse"] = rmse(real, fake)
                row["windows"] = len(keep)
                results[f"{name}/{subject}/{part}/{model_name}"] = row
                print(f"{name} {subject} {part} {model_name}: mae_hr {row['mae_hr']:.2f} corr {row['hr_corr']:.2f} "
                      f"real hr {row['hr_real_mean']:.0f} fake hr {row['hr_fake_mean']:.0f}", flush=True)
        # The rebuilt test windows must be the ones in the .npy files
        stored = len(np.load(root / name / "ecg_test_8sec.npy", mmap_mode="r"))
        print(f"{name}: rebuilt {total_test} test windows, stored {stored}", flush=True)
        results[f"{name}/check"] = {"rebuilt_test_windows": total_test, "stored_test_windows": stored}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
