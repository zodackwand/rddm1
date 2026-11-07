# Test 1: does the model use the PPG? Same noise, three conditions: the real PPG, the PPG of another window, zeros.
# If the ECG hardly changes between conditions, the model ignores the condition.
import argparse
import json
import warnings

import numpy as np
import torch

from diag_common import add_model_args, device_name, generate, hr_stats, load_models, pick, rmse
from train import load_split


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--datasets", nargs="*", default=["WESAD", "DALIA"])
    parser.add_argument("--n", type=int, default=600, help="8 s test windows per dataset")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = device_name()
    models = load_models(args, device)
    results = {}
    for name in args.datasets:
        ecg, ppg, _ = pick(load_split(args.data_root, "test", [name], 8), args.n, args.seed)
        real = ecg.squeeze(1).numpy().astype(np.float64)
        conditions = {
            "real_ppg": ppg,
            "other_window_ppg": torch.roll(ppg, args.n // 2, dims=0),
            "zero_ppg": torch.zeros_like(ppg),
        }
        for model_name, model in models.items():
            fakes = {c: generate(model, x, device, args.seed) for c, x in conditions.items()}
            for cond, fake in fakes.items():
                row = hr_stats(real, fake)
                row["rmse_to_real_ecg"] = rmse(real, fake)
                # How far the ECG moves when only the condition changes
                row["rmse_to_fake_with_real_ppg"] = rmse(fakes["real_ppg"], fake)
                results[f"{name}/{model_name}/{cond}"] = row
                print(f"{name} {model_name} {cond}: {row}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
