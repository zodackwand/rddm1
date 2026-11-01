import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import DATASETS, ROOT

FS = 128
ROI_SIZE = 32
BLOCK_WINDOWS = 100  # about 13 minutes of 8 s windows
MAX_LAG = FS  # look up to 1 s after an R-peak


def peak_positions(roi):
    # ROI starts 16 samples before its R-peak; skip regions clipped by the window start
    starts = np.flatnonzero(np.diff(roi) > 0) + 1
    return starts + ROI_SIZE // 2


def block_lag(ecg_roi, ppg):
    # Average PPG slope after R-peaks: synchronized signals give one clear peak
    segments = []
    slope = np.diff(ppg, axis=1)
    for roi, s in zip(ecg_roi, slope):
        for peak in peak_positions(roi):
            if peak + MAX_LAG <= len(s):
                segments.append(s[peak:peak + MAX_LAG])
    if not segments:
        return None
    mean = np.mean(segments, axis=0)
    if mean.std() == 0:
        return None
    return mean.argmax() / FS, (mean.max() - mean.mean()) / mean.std()


def check(root, name, plots):
    d = root / name
    splits = json.load(open(d / "splits.json"))
    assert not set(splits["train"]) & set(splits["test"]), "subjects overlap"

    ecg = np.concatenate([np.load(d / f"ecg_{p}_8sec.npy") for p in ["train", "test"]])
    ppg = np.concatenate([np.load(d / f"ppg_{p}_8sec.npy") for p in ["train", "test"]])
    roi = np.concatenate([np.load(d / f"roi_{p}_8sec.npy") for p in ["train", "test"]])

    print(f"=== {name}: {len(splits['train'])} train / {len(splits['test'])} test subjects, {len(ecg)} windows")
    print(f"  nan: {np.isnan(ecg).sum() + np.isnan(ppg).sum()}")
    print(f"  ecg range: {ecg.min():.2f}..{ecg.max():.2f}, ppg range: {ppg.min():.2f}..{ppg.max():.2f}")
    print(f"  roi fraction: {roi.mean():.3f}, windows without peaks: {(roi.sum(axis=1) == 0).mean():.3%}")
    n_peaks = np.array([len(peak_positions(r)) for r in roi])
    print(f"  heart rate from peaks, bpm: median {np.median(n_peaks) * 60 / 8:.0f}, "
          f"p5 {np.percentile(n_peaks, 5) * 60 / 8:.0f}, p95 {np.percentile(n_peaks, 95) * 60 / 8:.0f}")

    results = [block_lag(roi[i:i + BLOCK_WINDOWS], ppg[i:i + BLOCK_WINDOWS]) for i in range(0, len(roi), BLOCK_WINDOWS)]
    results = [r for r in results if r is not None]
    if results:
        lags = np.array([r[0] for r in results])
        contrast = np.array([r[1] for r in results])
        print(f"  ecg->ppg lag, s: median {np.median(lags):.3f}, p10 {np.percentile(lags, 10):.3f}, "
              f"p90 {np.percentile(lags, 90):.3f}, in 0.1-0.5 s: {np.mean((lags >= 0.1) & (lags <= 0.5)):.0%}, "
              f"contrast: {np.median(contrast):.1f}, blocks: {len(results)}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ecg), size=min(4, len(ecg)), replace=False)
    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 2.2 * len(idx)), squeeze=False)
    t = np.arange(ecg.shape[1]) / FS
    for ax, i in zip(axes[:, 0], idx):
        ax.plot(t, ecg[i], lw=0.8, label="ECG")
        ax.plot(t, ppg[i], lw=0.8, label="PPG")
        ax.fill_between(t, -1, 1, where=roi[i] > 0, alpha=0.15, color="gray", label="ROI")
    axes[0, 0].set_title(name)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(plots / f"{name}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--plots", default="check_plots")
    args = parser.parse_args()

    plots = Path(args.plots)
    plots.mkdir(exist_ok=True)
    for name in args.datasets:
        check(Path(args.root), name, plots)


if __name__ == "__main__":
    main()
