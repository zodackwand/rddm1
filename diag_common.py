import numpy as np
import torch

from data import ROOT
from evaluate import CHUNK, FS, ecg_bpm, load_authors
from train import generate_ecg, load_rddm, load_split


def add_model_args(parser):
    parser.add_argument("--path", help="our checkpoint directory")
    parser.add_argument("--epoch", type=int, default=499)
    parser.add_argument("--authors", help="folder with the authors' .pth files")
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", required=True, help="output json")


def load_models(args, device):
    models = {}
    if args.authors:
        models["authors"] = load_authors(args.authors, device)
    if args.path:
        models["ours"] = load_rddm(args.path, args.epoch, device)
    if not models:
        raise SystemExit("give --path and/or --authors")
    return models


def generate(models, ppg, device, seed, batch=256):
    # ppg: (N, 1, L) with L a multiple of 4 s; every 4 s chunk is generated on its own, as in evaluate.py
    torch.manual_seed(seed)
    out = []
    for part in torch.split(ppg, batch):
        halves = [generate_ecg(*models, h.to(device)).cpu() for h in torch.split(part, CHUNK, dim=-1)]
        out.append(torch.cat(halves, dim=-1))
    return torch.cat(out).squeeze(1).numpy().astype(np.float64)


def hr_stats(real, fake):
    # Heart rate as in evaluate.py: real ECG is already cleaned, fake ECG is cleaned first
    hr_real = np.array([ecg_bpm(x, clean=False) for x in real])
    hr_fake = np.array([ecg_bpm(x, clean=True) for x in fake])
    keep = (hr_fake != -1) & (hr_real != -1)
    err = np.abs(hr_real[keep] - hr_fake[keep])
    return {
        "mae_hr": float(err.mean()),
        "median_err": float(np.median(err)),
        "frac_err_over_10": float((err > 10).mean()),
        "hr_corr": float(np.corrcoef(hr_real[keep], hr_fake[keep])[0, 1]),
        "hr_found": float(keep.mean()),
        "hr_fake_mean": float(hr_fake[keep].mean()),
        "hr_real_mean": float(hr_real[keep].mean()),
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def pick(dataset, n, seed):
    # n windows chosen at random but reproducibly: (ecg, ppg, roi) tensors
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)[:n]
    return [t[idx] for t in dataset.tensors]


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"
