# Test 7: distribution of the real heart rate in train and test, per dataset and pooled.
# Test windows with a heart rate that train rarely has are where a memorising model has nothing to copy.
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from data import DATASETS, ROOT
from evaluate import ecg_bpm

BINS = np.arange(30, 211, 10)


def window_hr(window):
    warnings.filterwarnings("ignore")
    return ecg_bpm(window, clean=False)  # the stored real ECG is already cleaned


def sample_hr(pool, folder, part, n, seed):
    ecg = np.load(folder / f"ecg_{part}_8sec.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(seed).choice(len(ecg), size=min(n, len(ecg)), replace=False))
    hr = np.array(list(pool.map(window_hr, [np.asarray(ecg[i], dtype=np.float64) for i in idx], chunksize=64)))
    return hr[hr > 0]


def histogram(hr):
    counts, _ = np.histogram(hr, BINS)
    return (counts / max(counts.sum(), 1)).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--n", type=int, default=2000, help="8 s windows per dataset and part")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    results, pooled = {}, {"train": [], "test": []}
    with ProcessPoolExecutor(workers) as pool:
        for name in args.datasets:
            folder = Path(args.data_root) / name
            hr = {part: sample_hr(pool, folder, part, args.n, args.seed) for part in ["train", "test"]}
            for part in hr:
                pooled[part].append(hr[part])
            train_hist = np.array(histogram(hr["train"]))
            test_counts, _ = np.histogram(hr["test"], BINS)
            # Share of test windows that fall in a bin holding under 2% of this dataset's train windows
            rare = float(test_counts[train_hist < 0.02].sum() / max(test_counts.sum(), 1))
            results[name] = {
                "bins": BINS.tolist(), "train_hist": train_hist.tolist(), "test_hist": histogram(hr["test"]),
                "train_mean": float(hr["train"].mean()), "test_mean": float(hr["test"].mean()),
                "train_std": float(hr["train"].std()), "test_std": float(hr["test"].std()),
                "test_share_in_rare_train_bins": rare,
            }
            print(f"{name}: train hr {hr['train'].mean():.0f}±{hr['train'].std():.0f}, test hr {hr['test'].mean():.0f}±{hr['test'].std():.0f}, "
                  f"test windows in rare train bins {rare:.0%}", flush=True)

    train_all = np.concatenate(pooled["train"])
    train_hist = np.array(histogram(train_all))
    for name, test_hr in zip(args.datasets, pooled["test"]):
        counts, _ = np.histogram(test_hr, BINS)
        rare = float(counts[train_hist < 0.02].sum() / max(counts.sum(), 1))
        results[name]["test_share_in_rare_pooled_train_bins"] = rare
        print(f"{name}: test windows in rare bins of the pooled train {rare:.0%}", flush=True)
    results["pooled_train_hist"] = train_hist.tolist()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
