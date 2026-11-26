"""Random PPG+ECG pairs from every dataset: raw (original Hz, no filters) and after our preprocessing.

One window per subject, up to --n subjects per dataset. Output is a small JSON for the comparison pages.
Usage: sbatch sample_pairs.slurm   (writes sample_pairs.json, 4 s windows, 15 subjects per dataset)
       sbatch sample_pairs.slurm --all --seconds 8 --raw-only --out subjects_raw.json   (every subject)
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from data import DATASETS, ROOT
from preprocess import FILTER_ORDER, FLAT_RANGE, FS, LOADERS, SEED, process_window, resample, zscore

def to_list(x, digits=4):
    # NaN is kept as None so the raw signal is shown as recorded
    return [None if not np.isfinite(v) else round(float(v), digits) for v in x]


def sample_subject(job):
    name, subject, args, seed, split, seconds, raw_only = job
    warnings.filterwarnings("ignore")
    try:
        ecg_raw, ecg_fs, ppg_raw, ppg_fs = LOADERS[name][1](*args)
        ecg_raw, ppg_raw = np.asarray(ecg_raw, float), np.asarray(ppg_raw, float)
        # Same steps as preprocess.process_subject, on the whole recording
        ecg, ppg = resample(ecg_raw, ecg_fs), resample(ppg_raw, ppg_fs)
        n = min(len(ecg), len(ppg))
        ecg = sosfiltfilt(butter(FILTER_ORDER, 0.5, "highpass", fs=FS, output="sos"), ecg[:n])
        ppg = sosfiltfilt(butter(FILTER_ORDER, [0.5, 8], "bandpass", fs=FS, output="sos"), ppg[:n])
        ecg, ppg = zscore(ecg), zscore(ppg)

        length = seconds * FS
        starts = np.arange(0, n - length + 1, length)  # the same non-overlapping grid as preprocess.py
        rng = np.random.default_rng(seed)
        for start in rng.permutation(starts):
            ecg_w, ppg_w = ecg[start:start + length], ppg[start:start + length]
            if np.ptp(ecg_w) < FLAT_RANGE or np.ptp(ppg_w) < FLAT_RANGE:
                continue  # dropped by preprocess.py too
            t0 = start / FS
            e0, p0 = int(round(t0 * ecg_fs)), int(round(t0 * ppg_fs))
            row = {
                "subject": subject, "split": split, "start_s": round(float(t0), 2),
                "duration_s": round(n / FS, 1), "ecg_fs": float(ecg_fs), "ppg_fs": float(ppg_fs),
                "ecg_raw": to_list(ecg_raw[e0:e0 + int(round(seconds * ecg_fs))]),
                "ppg_raw": to_list(ppg_raw[p0:p0 + int(round(seconds * ppg_fs))]),
            }
            if not raw_only:
                ecg_p, ppg_p, _ = process_window(ecg_w, ppg_w)
                row["ecg"], row["ppg"] = to_list(ecg_p), to_list(ppg_p)
            return row
        return {"subject": subject, "split": split, "error": "no usable window"}
    except Exception as e:
        return {"subject": subject, "split": split, "error": f"{type(e).__name__}: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--n", type=int, default=15, help="pairs per dataset, one per subject")
    parser.add_argument("--out", default="sample_pairs.json")
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="every subject; failed ones are kept with their error")
    parser.add_argument("--raw-only", action="store_true", help="skip the processed version")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    args = parser.parse_args()
    root = Path(args.root)

    result = {"fs": FS, "seconds": args.seconds, "seed": SEED, "datasets": {}}
    for d_i, name in enumerate(args.datasets):
        subjects = LOADERS[name][0](root)
        split_file = root / name / "splits.json"
        splits = json.load(open(split_file)) if split_file.exists() else {"train": [], "test": []}
        part = {s: p for p in ["train", "test"] for s in splits[p]}
        rng = np.random.default_rng(SEED + d_i)
        # A few spare subjects in case some recordings fail to load
        order = range(len(subjects)) if args.all else rng.permutation(len(subjects))[:args.n + 5]
        picked = [subjects[i] for i in order]
        jobs = [(name, s, a, SEED * 1000 + d_i * 100 + j, part.get(s, "unused"), args.seconds, args.raw_only)
                for j, (s, a) in enumerate(picked)]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(sample_subject, jobs))
        for r in rows:
            if "error" in r:
                print(f"  {name} {r['subject']}: {r['error']}", flush=True)
        if not args.all:
            rows = [r for r in rows if "error" not in r][:args.n]
        result["datasets"][name] = rows
        print(f"{name}: {sum('error' not in r for r in rows)} pairs, {len(rows)} subjects", flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"saved {args.out}, {os.path.getsize(args.out) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
