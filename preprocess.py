import argparse
import glob
import json
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, resample_poly, sosfiltfilt

from data import DATASETS, ROOT

FS = 128
WINDOW_SECONDS = [4, 8]
ROI_SIZE = 32
TEST_FRACTION = 0.2
SEED = 31
FILTER_ORDER = 4  # not stated in the paper
FLAT_RANGE = 1e-3  # in z-score units: below this a window is a flat line


# Every lister returns (subject_id, loader_args); loaders return (ecg, ecg_fs, ppg, ppg_fs)
def list_bidmc(root):
    folder = root / "BIDMC/raw/bidmc-ppg-and-respiration-dataset-1.0.0"
    headers = sorted(glob.glob(str(folder / "bidmc[0-9][0-9].hea")))
    return [(os.path.basename(h)[:-4], (h,)) for h in headers]


def load_bidmc(header):
    with open(header) as f:
        lines = [line.strip() for line in f]
    n_channels = int(lines[0].split()[1])
    fs = float(lines[0].split()[2])

    gains, baselines, names = [], [], []
    for line in lines[1:n_channels + 1]:
        parts = line.split()
        gain, baseline = parts[2].split("/")[0].rstrip(")").split("(")
        gains.append(float(gain))
        baselines.append(float(baseline))
        names.append(parts[-1].rstrip(","))

    if "II" not in names or "PLETH" not in names:
        raise ValueError(f"missing channels, found {names}")

    raw = np.fromfile(header[:-4] + ".dat", dtype="<i2").reshape(-1, n_channels)

    def channel(name):
        i = names.index(name)
        return (raw[:, i] - baselines[i]) / gains[i]

    return channel("II"), fs, channel("PLETH"), fs


def list_capno(root):
    files = sorted(glob.glob(str(root / "CAPNO/raw/data/csv/*_8min_signal.csv")))
    return [(os.path.basename(f)[:4], (f,)) for f in files]


def load_capno(path):
    df = pd.read_csv(path, usecols=["ecg_y", "pleth_y"])
    return df["ecg_y"].to_numpy(float), 300, df["pleth_y"].to_numpy(float), 300


def list_pkl(root, pattern):
    files = sorted(glob.glob(str(root / pattern)))
    return [(os.path.basename(f)[:-4], (f,)) for f in files]


def load_pkl(path):
    with open(path, "rb") as f:
        signal = pickle.load(f, encoding="latin1")["signal"]
    return signal["chest"]["ECG"][:, 0], 700, signal["wrist"]["BVP"][:, 0], 64


def list_mimic(root):
    subjects = []
    for group in ["af", "non_af"]:
        path = root / f"MIMIC-AFib/raw/mimic_perform_{group}_data.mat"
        data = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
        subjects += [(f"{group}_{i:03d}", (str(path), i)) for i in range(len(data))]
    return subjects


def load_mimic(path, index):
    record = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)["data"][index]
    return (
        np.asarray(record.ekg.v, dtype=float), float(record.ekg.fs),
        np.asarray(record.ppg.v, dtype=float), float(record.ppg.fs),
    )


LOADERS = {
    "BIDMC": (list_bidmc, load_bidmc),
    "CAPNO": (list_capno, load_capno),
    "DALIA": (lambda root: list_pkl(root, "DALIA/raw/PPG_FieldStudy/S*/S*.pkl"), load_pkl),
    "MIMIC-AFib": (list_mimic, load_mimic),
    "WESAD": (lambda root: list_pkl(root, "WESAD/raw/WESAD/S*/S*.pkl"), load_pkl),
}


def resample(x, fs):
    ratio = Fraction(FS / fs).limit_denominator(10000)
    return resample_poly(np.nan_to_num(x), ratio.numerator, ratio.denominator)


def zscore(x):
    return (x - x.mean()) / x.std()


def minmax(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def make_roi(ecg):
    # Same ROI as the authors' ECGDataset
    _, info = nk.ecg_peaks(ecg, sampling_rate=FS, method="pantompkins1985", correct_artifacts=True, show=False)
    roi = np.zeros_like(ecg)
    for peak in info["ECG_R_Peaks"]:
        roi_start = max(0, int(peak) - ROI_SIZE // 2)
        roi[roi_start:min(roi_start + ROI_SIZE, len(ecg))] = 1
    return roi


def process_window(ecg, ppg):
    # Same steps and order as the authors' get_datasets and ECGDataset
    ecg, ppg = minmax(ecg), minmax(ppg)
    ppg = nk.ppg_clean(ppg, sampling_rate=FS)
    ecg = nk.ecg_clean(ecg, sampling_rate=FS, method="pantompkins1985")
    return ecg, ppg, make_roi(ecg)


def process_subject(job):
    # job: (dataset, subject, loader args[, offset in seconds, window lengths]); the extras are for shifted windows
    name, subject, args, *extra = job
    offset, window_seconds = extra if extra else (0, WINDOW_SECONDS)
    warnings.filterwarnings("ignore")
    out = {"name": name, "subject": subject, "windows": {}, "dropped": 0, "error": None}
    try:
        ecg, ecg_fs, ppg, ppg_fs = LOADERS[name][1](*args)
        if np.nanmax(ecg) == np.nanmin(ecg) or np.nanmax(ppg) == np.nanmin(ppg):
            raise ValueError("constant signal")
        ecg, ppg = resample(ecg, ecg_fs), resample(ppg, ppg_fs)
        n = min(len(ecg), len(ppg))

        ecg = sosfiltfilt(butter(FILTER_ORDER, 0.5, "highpass", fs=FS, output="sos"), ecg[:n])
        ppg = sosfiltfilt(butter(FILTER_ORDER, [0.5, 8], "bandpass", fs=FS, output="sos"), ppg[:n])
        ecg, ppg = zscore(ecg), zscore(ppg)

        for seconds in window_seconds:
            length = seconds * FS
            rows = []
            for start in range(offset * FS, n - length + 1, length):
                ecg_w, ppg_w = ecg[start:start + length], ppg[start:start + length]
                # Flat windows have no min-max range and no R-peaks
                if np.ptp(ecg_w) < FLAT_RANGE or np.ptp(ppg_w) < FLAT_RANGE:
                    out["dropped"] += 1
                    continue
                try:
                    rows.append(process_window(ecg_w, ppg_w))
                except Exception:
                    out["dropped"] += 1
            if rows:
                out["windows"][seconds] = np.asarray(rows, dtype="float32")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def split_subjects(subjects):
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sorted(subjects)).tolist()
    n_test = max(1, round(len(shuffled) * TEST_FRACTION))
    return shuffled[n_test:], shuffled[:n_test]


def main_shifted(args, root):
    # Train windows of 4 s that start `offset` seconds later: the same subjects and the same preprocessing,
    # saved next to the originals as *_train_4sec_off<offset>.npy. The split is read from splits.json, never redrawn.
    for name in args.datasets:
        train = json.load(open(root / name / "splits.json"))["train"]
        listing = dict(LOADERS[name][0](root))
        jobs = [(name, s, listing[s], args.offset, [4]) for s in train]
        print(f"=== {name}: {len(jobs)} train subjects, offset {args.offset} s", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(process_subject, jobs))
        for r in results:
            if r["error"]:
                print(f"  skipped {r['subject']}: {r['error']}", flush=True)
        data = np.concatenate([r["windows"][4] for r in results if 4 in r["windows"]])
        for i, key in enumerate(["ecg", "ppg", "roi"]):
            np.save(root / name / f"{key}_train_4sec_off{args.offset}.npy", data[:, i])
        print(f"  train 4s offset {args.offset}: {data.shape[0]} windows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    parser.add_argument("--max-subjects", type=int, default=None, help="use only for a quick test")
    parser.add_argument("--offset", type=int, default=0, help="seconds; above 0 builds only shifted train windows")
    args = parser.parse_args()

    root = Path(args.root)
    if args.offset:
        return main_shifted(args, root)

    for name in args.datasets:
        subjects = LOADERS[name][0](root)[:args.max_subjects]
        print(f"=== {name}: {len(subjects)} subjects", flush=True)
        if not subjects:
            raise RuntimeError(f"{name}: no subjects found")

        jobs = [(name, subject, loader_args) for subject, loader_args in subjects]
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(process_subject, jobs))

        results = {r["subject"]: r for r in results}
        for r in results.values():
            if r["error"]:
                print(f"  skipped {r['subject']}: {r['error']}", flush=True)
        usable = {s: r for s, r in results.items() if r["windows"]}
        if not usable:
            raise RuntimeError(f"{name}: no usable subjects")

        train, test = split_subjects(usable)
        assert not set(train) & set(test)
        out_dir = root / name
        for seconds in WINDOW_SECONDS:
            for part, ids in [("train", train), ("test", test)]:
                arrays = [usable[s]["windows"][seconds] for s in ids if seconds in usable[s]["windows"]]
                data = np.concatenate(arrays)  # (windows, [ecg, ppg, roi], length)
                np.save(out_dir / f"ecg_{part}_{seconds}sec.npy", data[:, 0])
                np.save(out_dir / f"ppg_{part}_{seconds}sec.npy", data[:, 1])
                np.save(out_dir / f"roi_{part}_{seconds}sec.npy", data[:, 2])
                print(f"  {part} {seconds}s: {data.shape[0]} windows, {len(ids)} subjects", flush=True)

        dropped = sum(r["dropped"] for r in results.values())
        print(f"  dropped windows: {dropped}", flush=True)
        with open(out_dir / "splits.json", "w") as f:
            json.dump({"seed": SEED, "train": train, "test": test}, f, indent=2)


if __name__ == "__main__":
    main()
