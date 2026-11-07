# Test 2: heart-rate quality over training epochs, on test and on train windows.
# A rising gap between train and test means memorising subjects; a flat curve means the PPG is never learned.
import argparse
import glob
import json
import re
import warnings

import numpy as np
import torch

from diag_common import device_name, generate, hr_stats, pick, rmse
from data import ROOT
from train import load_rddm, load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="our checkpoint directory")
    parser.add_argument("--epochs", type=int, nargs="*", help="default: every RDDM_epoch*.pth found")
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=["WESAD", "DALIA"])
    parser.add_argument("--n", type=int, default=600, help="8 s windows per dataset and part")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    epochs = args.epochs or sorted(int(re.search(r"epoch(\d+)", p).group(1)) for p in glob.glob(f"{args.path}/RDDM_epoch*.pth"))
    print("epochs", epochs, flush=True)
    device = device_name()
    data = {
        (name, part): pick(load_split(args.data_root, part, [name], 8), args.n, args.seed)
        for name in args.datasets for part in ["train", "test"]
    }

    results = {}
    for epoch in epochs:
        models = load_rddm(args.path, epoch, device)
        for (name, part), (ecg, ppg, _) in data.items():
            real = ecg.squeeze(1).numpy().astype(np.float64)
            fake = generate(models, ppg, device, args.seed)
            row = hr_stats(real, fake)
            row["rmse"] = rmse(real, fake)
            results[f"epoch{epoch}/{name}/{part}"] = row
            print(f"epoch {epoch} {name} {part}: mae_hr {row['mae_hr']:.2f} corr {row['hr_corr']:.2f} rmse {row['rmse']:.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
