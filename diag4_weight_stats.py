# Test 4: weight statistics of our networks against the authors', layer by layer. CPU only.
# Looks for layers that stayed small or moved little, above all the cross-attention that carries the PPG into the denoiser.
import argparse
import json

import torch

from evaluate import load_authors
from train import load_rddm


def named_params(models):
    rddm, cond_net1, cond_net2 = models
    for prefix, module in [("rddm", rddm), ("cond1", cond_net1), ("cond2", cond_net2)]:
        for name, p in module.named_parameters():
            yield f"{prefix}.{name}", p.detach().double()


def group(name):
    if "cross_attention." in name and "in_proj_weight" in name:
        return "cross-attention q/k/v projection"
    if "cross_attention.out_proj.weight" in name:
        return "cross-attention output projection"
    if ".ff_cross." in name and name.endswith("weight"):
        return "cross-attention feed-forward / norm"
    if name.endswith("weight") and (".double_conv.0." in name or ".double_conv.3." in name):
        return "conv"
    if ".double_conv.1" in name or ".double_conv.4" in name:
        return "group norm"
    if "up" in name and name.endswith("weight") and "double_conv" not in name and "cross" not in name:
        return "transposed conv"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--epoch", type=int, default=499)
    parser.add_argument("--authors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ours = dict(named_params(load_rddm(args.path, args.epoch)))
    authors = dict(named_params(load_authors(args.authors, "cpu")))
    assert ours.keys() == authors.keys(), "parameter names differ"

    rows, groups = [], {}
    for name in ours:
        a, b = ours[name], authors[name]
        # RMS per element: independent of the layer size
        rms_ours, rms_auth = a.pow(2).mean().sqrt().item(), b.pow(2).mean().sqrt().item()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        rows.append({"name": name, "rms_ours": rms_ours, "rms_authors": rms_auth,
                     "ratio": rms_ours / rms_auth if rms_auth else None, "cosine": cos, "numel": a.numel()})
        g = groups.setdefault(group(name), {"ratios": [], "numel": 0})
        if rms_auth:
            g["ratios"].append(rms_ours / rms_auth)
        g["numel"] += a.numel()

    summary = {
        k: {"layers": len(v["ratios"]), "numel": v["numel"],
            "ratio_median": sorted(v["ratios"])[len(v["ratios"]) // 2],
            "ratio_min": min(v["ratios"]), "ratio_max": max(v["ratios"])}
        for k, v in groups.items() if v["ratios"]
    }
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nlargest deviations of ours from the authors' RMS (ratio ours/authors):")
    for r in sorted((r for r in rows if r["ratio"]), key=lambda r: abs(torch.log(torch.tensor(r["ratio"])).item()), reverse=True)[:25]:
        print(f"  {r['name']}: ratio {r['ratio']:.2f} (ours {r['rms_ours']:.4f}, authors {r['rms_authors']:.4f})")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "layers": rows}, f, indent=2)


if __name__ == "__main__":
    main()
