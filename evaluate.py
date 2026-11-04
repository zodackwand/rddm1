import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import torch
from biosppy.signals import ecg as ecg_func
from biosppy.signals import tools
from torch.utils.data import DataLoader

from data import DATASETS, ROOT
from train import ConditionNet, build_rddm, generate_ecg, load_rddm, load_split

FS = 128
CHUNK = 4 * FS  # the model always generates 4 s windows
HR_DATASETS = ["WESAD", "DALIA"]  # Table 2 of the paper


# Metrics are copied from the authors' metrics.py so the numbers are comparable
def calculate_fd(true_ecg, fake_ecg):
    def stats(x):
        x = x.reshape(-1, CHUNK)
        return x.mean(axis=0), np.cov(x, rowvar=False)

    mu1, sigma1 = stats(true_ecg)
    mu2, sigma2 = stats(fake_ecg)
    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def ecg_bpm(signal, clean):
    try:
        return _ecg_bpm(signal, clean)
    except Exception:
        return -1.0


def _ecg_bpm(signal, clean):
    if clean:
        signal = nk.ecg_clean(signal, sampling_rate=FS, method="pantompkins1985")
    rpeaks, = ecg_func.hamilton_segmenter(signal=signal, sampling_rate=FS)
    rpeaks, = ecg_func.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=FS, tol=0.05)
    _, rpeaks = ecg_func.extract_heartbeats(signal=signal, rpeaks=rpeaks, sampling_rate=FS, before=0.2, after=0.4)
    if len(rpeaks) < 2:
        return -1.0
    _, hr = tools.get_heart_rate(beats=rpeaks, sampling_rate=FS, smooth=True, size=3)
    return float(np.mean(hr)) if len(hr) else -1.0


def mae_hr(real_ecg, fake_ecg):
    real = np.array([ecg_bpm(x, clean=False) for x in real_ecg])
    fake = np.array([ecg_bpm(x, clean=True) for x in fake_ecg])
    # As in the authors' code: only windows where the fake ECG gave a heart rate
    keep = fake != -1
    return float(np.mean(np.abs(real[keep] - fake[keep]))), float(keep.mean())


def load_authors(folder, device):
    rddm, cond_net1, cond_net2 = build_rddm().to(device), ConditionNet().to(device), ConditionNet().to(device)
    rddm.load_state_dict(torch.load(f"{folder}/rddm_main_network.pth", map_location=device))
    cond_net1.load_state_dict(torch.load(f"{folder}/rddm_condition_encoder_1.pth", map_location=device))
    cond_net2.load_state_dict(torch.load(f"{folder}/rddm_condition_encoder_2.pth", map_location=device))
    return rddm.eval(), cond_net1.eval(), cond_net2.eval()


def run(models, dataset, batch_size, gen_batch, device, seed):
    # Same order as the authors: shuffled batches of 512, FD averaged over batches
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    reals, fakes, fds = [], [], []
    for y_ecg, x_ppg, _ in loader:
        parts = []
        for ppg in torch.split(x_ppg, gen_batch):
            # 8 s windows are generated as two 4 s halves, as in std_eval.py
            halves = [generate_ecg(*models, h.to(device)).cpu() for h in torch.split(ppg, CHUNK, dim=-1)]
            parts.append(torch.cat(halves, dim=-1))
        fake = torch.cat(parts).squeeze(1).numpy().astype(np.float64)
        real = y_ecg.squeeze(1).numpy().astype(np.float64)
        fds.append(calculate_fd(real, fake))
        reals.append(real)
        fakes.append(fake)
    return np.concatenate(reals), np.concatenate(fakes), float(np.mean(fds))


def plot(real, fake, path, title):
    idx = np.random.default_rng(0).choice(len(real), size=min(4, len(real)), replace=False)
    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 2.2 * len(idx)), squeeze=False)
    t = np.arange(real.shape[1]) / FS
    for ax, i in zip(axes[:, 0], idx):
        ax.plot(t, real[i], lw=0.8, label="real")
        ax.plot(t, fake[i], lw=0.8, label="generated")
    axes[0, 0].set_title(title)
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--path", help="our checkpoint directory")
    parser.add_argument("--epoch", type=int, default=499)
    parser.add_argument("--authors", help="folder with the authors' .pth files instead of --path")
    parser.add_argument("--out", required=True, help="folder for metrics.json and plots")
    parser.add_argument("--batch-size", type=int, default=512, help="FD depends on it, keep 512 as the authors")
    parser.add_argument("--gen-batch", type=int, default=512, help="lower it on CUDA out of memory")
    parser.add_argument("--part", default="test", help="train only to diagnose overfitting")
    parser.add_argument("--max-windows", type=int, default=None, help="use only for a quick test")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.authors:
        models = load_authors(args.authors, device)
    else:
        models = load_rddm(args.path, args.epoch, device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.datasets:
        results[name] = {}
        for seconds in [4, 8] if name in HR_DATASETS else [4]:
            dataset = load_split(args.data_root, args.part, [name], seconds)
            if args.max_windows:
                dataset = torch.utils.data.Subset(dataset, range(min(args.max_windows, len(dataset))))
            real, fake, fd = run(models, dataset, args.batch_size, args.gen_batch, device, args.seed)
            if seconds == 4:
                # Table 1: RMSE and FD on 4 s windows
                results[name]["rmse"] = float(np.sqrt(np.mean((fake - real) ** 2)))
                results[name]["fd"] = fd
            else:
                # Table 2: heart rate on 8 s windows
                results[name]["mae_hr"], results[name]["hr_found"] = mae_hr(real, fake)
            results[name][f"windows_{seconds}s"] = len(real)
            np.savez(out / f"{name}_{seconds}sec.npz", real=real.astype(np.float32), fake=fake.astype(np.float32))
            plot(real, fake, out / f"{name}_{seconds}sec.png", f"{name}, {seconds} s")
            print(name, seconds, results[name], flush=True)

    with open(out / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
