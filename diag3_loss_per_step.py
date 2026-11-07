# Test 3: training losses of each model at every diffusion step t, with the real PPG and with zeros.
# The gap between the two is how much the condition helps denoising. Compare ours with the authors' and train with test.
import argparse
import json
import warnings

import torch
import torch.nn.functional as F

from diag_common import add_model_args, device_name, load_models, pick
from train import load_split


@torch.no_grad()
def losses_at_t(models, y, ppg, roi, t_int, seed, device):
    rddm, cond_net1, cond_net2 = models
    torch.manual_seed(seed)  # the same noise for every model and condition
    t = torch.full((y.shape[0],), t_int, device=device, dtype=torch.long)
    eps, unmasked_eps = rddm.create_noise_in_regions(roi)
    sqrtab, sqrtmab = rddm.sqrtab[t, None, None], rddm.sqrtmab[t, None, None]
    x_t = sqrtab * y + sqrtmab * eps
    x_t_unmasked = sqrtab * y + sqrtmab * unmasked_eps
    pred_x_t = rddm.region_model(x_t_unmasked, cond_net2(ppg), t / rddm.n_T)
    pred_eps = rddm.eps_model(x_t, cond_net1(ppg), t / rddm.n_T)
    return F.mse_loss(pred_eps, eps).item(), F.mse_loss(pred_x_t, x_t).item()


def main():
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--datasets", nargs="*", default=["WESAD", "DALIA"])
    parser.add_argument("--n", type=int, default=2048, help="4 s windows per dataset and part")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    device = device_name()
    models = load_models(args, device)
    n_T = next(iter(models.values()))[0].n_T
    results = {}
    for name in args.datasets:
        for part in ["train", "test"]:
            ecg, ppg, roi = pick(load_split(args.data_root, part, [name], 4), args.n, args.seed)
            for model_name, model in models.items():
                for cond in ["real_ppg", "zero_ppg"]:
                    # t runs over 1..n_T-1 as in training
                    for t_int in range(1, n_T):
                        sums = [0.0, 0.0]
                        for y, x, r in zip(ecg.split(args.batch), ppg.split(args.batch), roi.split(args.batch)):
                            x = x if cond == "real_ppg" else torch.zeros_like(x)
                            d, g = losses_at_t(model, y.to(device), x.to(device), r.to(device), t_int, args.seed, device)
                            sums[0] += d * len(y)
                            sums[1] += g * len(y)
                        results[f"{name}/{part}/{model_name}/{cond}/t{t_int}"] = {
                            "ddpm": sums[0] / len(ecg), "region": sums[1] / len(ecg),
                        }
                    row = [results[f"{name}/{part}/{model_name}/{cond}/t{t}"]["ddpm"] for t in range(1, n_T)]
                    print(f"{name} {part} {model_name} {cond} ddpm per t: " + " ".join(f"{v:.4f}" for v in row), flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
