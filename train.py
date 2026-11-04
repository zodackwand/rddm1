import argparse
import contextlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from data import DATASETS, ROOT


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling without skip connection: used in ConditionNet
class SegmentUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        return self.conv(self.up(x1))

# Upsampling with skip connection: used in the denoising U-Net
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels * 2, out_channels)  # *2 because of the concat

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ConditionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc_c = DoubleConv(1, 64)
        self.inc_freq = DoubleConv(1, 64)  # unused, kept to match authors' checkpoints

        self.down1_c = Down(64, 128)
        self.down2_c = Down(128, 256)
        self.down3_c = Down(256, 512)
        self.down4_c = Down(512, 1024)
        self.down5_c = Down(1024, 1024)

        self.up1_c = SegmentUp(1024, 512)
        self.up2_c = SegmentUp(512, 256)
        self.up3_c = SegmentUp(256, 128)
        self.up4_c = SegmentUp(128, 64)
        self.up5_c = SegmentUp(64, 32)

    def forward(self, x):
        d1 = self.inc_c(x)
        d2 = self.down1_c(d1)
        d3 = self.down2_c(d2)
        d4 = self.down3_c(d3)
        d5 = self.down4_c(d4)
        d6 = self.down5_c(d5)

        u1 = self.up1_c(d6)
        u2 = self.up2_c(u1)
        u3 = self.up3_c(u2)
        u4 = self.up4_c(u3)
        u5 = self.up5_c(u4)

        return {
            "down_conditions": [d1, d2, d3, d4, d5, d6],
            "up_conditions": [u1, u2, u3, u4, u5],
        }


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([embed_dim])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([embed_dim]),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, c):
        # (batch, channels, length) -> (batch, length, channels): attention runs over time steps
        x_ln = self.ln(x.permute(0, 2, 1))
        c_ln = self.ln(c.permute(0, 2, 1))  # same LayerNorm for x and c, as in the original
        attention_value, _ = self.cross_attention(x_ln, c_ln, c_ln)
        attention_value = attention_value + x_ln  # residual from normalized x, as in the original
        attention_value = self.ff_cross(attention_value) + attention_value
        return attention_value.permute(0, 2, 1)


# This network takes ECG x with noise, takes c from cross attention and step t. On every step we add encoding of t, then add PPG map using cross attention.

class DiffusionUNetCrossAttention(nn.Module):
    def __init__(self, channels=1, num_heads=8):
        super().__init__()
        self.channels = channels

        self.inc_x = DoubleConv(channels, 64)
        self.inc_freq = DoubleConv(channels, 64)  # unused, kept to match authors' checkpoints

        self.down1_x = Down(64, 128)
        self.down2_x = Down(128, 256)
        self.down3_x = Down(256, 512)
        self.down4_x = Down(512, 1024)
        self.down5_x = Down(1024, 1024)

        self.up1_x = Up(1024, 512)
        self.up2_x = Up(512, 256)
        self.up3_x = Up(256, 128)
        self.up4_x = Up(128, 64)
        self.up5_x = Up(64, 32)

        self.cross_attention_down1 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_down2 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_down3 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_down4 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_down5 = CrossAttentionBlock(1024, num_heads)
        self.cross_attention_down6 = CrossAttentionBlock(1024, num_heads)

        self.cross_attention_up1 = CrossAttentionBlock(512, num_heads)
        self.cross_attention_up2 = CrossAttentionBlock(256, num_heads)
        self.cross_attention_up3 = CrossAttentionBlock(128, num_heads)
        self.cross_attention_up4 = CrossAttentionBlock(64, num_heads)
        self.cross_attention_up5 = CrossAttentionBlock(32, num_heads)

        self.outc_x = OutConv(32, channels)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1).repeat(1, 1, embed_size)

    def forward(self, x, c, t):
        t = t.unsqueeze(-1)  # (batch,) -> (batch, 1)
        down = c["down_conditions"]
        up = c["up_conditions"]

        # Downward path
        x1 = self.inc_x(x)
        x1 = self.cross_attention_down1(x1, down[0])
        length = x1.shape[-1]

        x2 = self.down1_x(x1) + self.pos_encoding(t, 128, length // 2)
        x2 = self.cross_attention_down2(x2, down[1])

        x3 = self.down2_x(x2) + self.pos_encoding(t, 256, length // 4)
        x3 = self.cross_attention_down3(x3, down[2])

        x4 = self.down3_x(x3) + self.pos_encoding(t, 512, length // 8)
        x4 = self.cross_attention_down4(x4, down[3])

        x5 = self.down4_x(x4) + self.pos_encoding(t, 1024, length // 16)
        x5 = self.cross_attention_down5(x5, down[4])

        x6 = self.down5_x(x5) + self.pos_encoding(t, 1024, length // 32)
        x6 = self.cross_attention_down6(x6, down[5])

        # Upward path
        x = self.up1_x(x6, x5) + self.pos_encoding(t, 512, length // 16)
        x = self.cross_attention_up1(x, up[0])

        x = self.up2_x(x, x4) + self.pos_encoding(t, 256, length // 8)
        x = self.cross_attention_up2(x, up[1])

        x = self.up3_x(x, x3) + self.pos_encoding(t, 128, length // 4)
        x = self.cross_attention_up3(x, up[2])

        x = self.up4_x(x, x2) + self.pos_encoding(t, 64, length // 2)
        x = self.cross_attention_up4(x, up[3])

        x = self.up5_x(x, x1) + self.pos_encoding(t, 32, length)
        x = self.cross_attention_up5(x, up[4])

        return self.outc_x(x)


def ddpm_schedule(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,                    # alpha_t
        "oneover_sqrta": oneover_sqrta,        # 1 / sqrt(alpha_t)
        "sqrt_beta_t": sqrt_beta_t,            # sqrt(beta_t)
        "alphabar_t": alphabar_t,              # alpha_bar_t
        "sqrtab": sqrtab,                      # sqrt(alpha_bar_t): signal weight
        "sqrtmab": sqrtmab,                    # sqrt(1 - alpha_bar_t): noise weight
        "mab_over_sqrtmab": mab_over_sqrtmab,  # (1 - alpha_t) / sqrt(1 - alpha_bar_t)
        "beta_t": beta_t,
    }


class RDDM(nn.Module):
    def __init__(self, eps_model, region_model, betas, n_T, criterion=nn.MSELoss()):
        super().__init__()
        self.eps_model = eps_model
        self.region_model = region_model
        self.n_T = n_T
        self.criterion = criterion

        # register_buffer moves the tensors with the model and saves them in state_dict
        for k, v in ddpm_schedule(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def create_noise_in_regions(self, patch_labels):
        patch_roi = torch.round(patch_labels)
        mask = patch_roi == 1
        random_noise = torch.randn_like(patch_roi)
        masked_noise = random_noise * mask.float()
        return masked_noise, random_noise

    def forward(self, x=None, cond1=None, cond2=None, mode="train", patch_labels=None, window_size=128 * 4):
        if mode == "train":
            # n_T is excluded, as in the original
            _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(x.device)

            eps, unmasked_eps = self.create_noise_in_regions(patch_labels)

            x_t = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * eps
            x_t_unmasked = self.sqrtab[_ts, None, None] * x + self.sqrtmab[_ts, None, None] * unmasked_eps

            pred_x_t = self.region_model(x_t_unmasked, cond2, _ts / self.n_T)
            pred_masked_eps = self.eps_model(x_t, cond1, _ts / self.n_T)

            ddpm_loss = self.criterion(eps, pred_masked_eps)
            region_loss = self.criterion(pred_x_t, x_t)

            return ddpm_loss, region_loss

        elif mode == "sample":
            n_sample = cond1["down_conditions"][-1].shape[0]
            device = cond1["down_conditions"][-1].device

            x_i = torch.randn(n_sample, 1, window_size).to(device)

            for i in range(self.n_T, 0, -1):
                z = torch.randn(n_sample, 1, window_size).to(device) if i > 1 else 0
                t = torch.tensor(i / self.n_T).to(device).repeat(n_sample)

                # Full noise -> noise only in ROI
                x_i = self.region_model(x_i, cond2, t)

                # Noise in ROI -> one step closer to ECG
                eps = self.eps_model(x_i, cond1, t)

                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )

            return x_i


def make_scheduler(optim, t_warmup, t_max):
    def lr_lambda(epoch):
        if epoch < t_warmup:
            return (epoch + 1) / t_warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - t_warmup) / (t_max - t_warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def load_split(root, part, datasets, seconds=4):
    arrays = {"ecg": [], "ppg": [], "roi": []}
    for name in datasets:
        for key in arrays:
            arrays[key].append(np.load(Path(root) / name / f"{key}_{part}_{seconds}sec.npy"))
    tensors = [torch.from_numpy(np.concatenate(arrays[key])).float().unsqueeze(1) for key in ["ecg", "ppg", "roi"]]
    return TensorDataset(*tensors)


def build_rddm(num_heads=8, n_T=10):
    return RDDM(
        eps_model=DiffusionUNetCrossAttention(num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(num_heads=num_heads),
        betas=(1e-4, 0.2),
        n_T=n_T,
    )


def load_rddm(checkpoint_dir, epoch, device="cpu"):
    rddm, cond_net1, cond_net2 = build_rddm().to(device), ConditionNet().to(device), ConditionNet().to(device)
    rddm.load_state_dict(torch.load(f"{checkpoint_dir}/RDDM_epoch{epoch}.pth", map_location=device))
    cond_net1.load_state_dict(torch.load(f"{checkpoint_dir}/ConditionNet1_epoch{epoch}.pth", map_location=device))
    cond_net2.load_state_dict(torch.load(f"{checkpoint_dir}/ConditionNet2_epoch{epoch}.pth", map_location=device))
    return rddm.eval(), cond_net1.eval(), cond_net2.eval()


@torch.no_grad()
def generate_ecg(rddm, cond_net1, cond_net2, ppg):
    # ppg: preprocessed tensor (batch, 1, 512) on the models' device
    return rddm(cond1=cond_net1(ppg), cond2=cond_net2(ppg), mode="sample", window_size=ppg.shape[-1])


class TrainStep(nn.Module):
    # One module for DDP: the condition nets run outside RDDM.forward
    def __init__(self, rddm, cond_net1, cond_net2):
        super().__init__()
        self.rddm = rddm
        self.cond_net1 = cond_net1
        self.cond_net2 = cond_net2

    def forward(self, y_ecg, x_ppg, ecg_roi):
        return self.rddm(x=y_ecg, cond1=self.cond_net1(x_ppg), cond2=self.cond_net2(x_ppg), patch_labels=ecg_roi)


def train_rddm(config, dataloader, sampler=None):
    device = config["device"]
    path = config["path"]
    rank, world = config["rank"], config["world"]
    main_process = rank == 0
    os.makedirs(path, exist_ok=True)

    rddm = build_rddm(config["attention_heads"], config["nT"]).to(device)
    cond_net1 = ConditionNet().to(device)
    cond_net2 = ConditionNet().to(device)
    step_model = TrainStep(rddm, cond_net1, cond_net2)

    optim = torch.optim.AdamW(step_model.parameters(), lr=config["lr"])
    scheduler = make_scheduler(optim, config["t_warmup"], config["n_epoch"])

    # Resume after a Slurm time limit
    start_epoch = 0
    last_path = f"{path}/last.pt"
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        rddm.load_state_dict(ckpt["rddm"])
        cond_net1.load_state_dict(ckpt["cond_net1"])
        cond_net2.load_state_dict(ckpt["cond_net2"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        if main_process:
            print(f"Resumed from epoch {ckpt['epoch']}", flush=True)

    if world > 1:
        device_ids = [config["local_rank"]] if device.startswith("cuda") else None
        # inc_freq layers exist in the authors' networks but forward never uses them
        step_model = nn.parallel.DistributedDataParallel(step_model, device_ids=device_ids, find_unused_parameters=True)

    for epoch in range(start_epoch, config["n_epoch"]):
        # Seeding per epoch makes a resumed run match an uninterrupted one
        torch.manual_seed(config["seed"] * 1000 + epoch * world + rank)
        if sampler is not None:
            sampler.set_epoch(epoch)
        step_model.train()
        lr = optim.param_groups[0]["lr"]
        totals = torch.zeros(3, device=device)  # ddpm, region, steps
        started = time.time()

        for y_ecg, x_ppg, ecg_roi in dataloader:
            batch = [t.to(device, non_blocking=True) for t in (y_ecg, x_ppg, ecg_roi)]
            optim.zero_grad()
            # Gradient accumulation: the batch is too large for one GPU at once
            micro_batches = list(zip(*[t.split(config["micro_batch"]) for t in batch]))
            for i, micro in enumerate(micro_batches):
                share = micro[0].shape[0] / batch[0].shape[0]
                last = i == len(micro_batches) - 1
                # DDP syncs gradients only on the last micro-batch
                sync = step_model.no_sync() if world > 1 and not last else contextlib.nullcontext()
                with sync:
                    ddpm_loss, region_loss = step_model(*micro)
                    loss = config["alpha1"] * ddpm_loss + config["alpha2"] * region_loss
                    (loss * share).backward()
                totals[0] += ddpm_loss.detach() * share
                totals[1] += region_loss.detach() * share
            optim.step()
            totals[2] += 1
            if config["max_steps"] and totals[2] >= config["max_steps"]:
                break

        if world > 1:
            dist.all_reduce(totals)
        ddpm_mean, region_mean = (totals[:2] / totals[2]).tolist()
        if not math.isfinite(ddpm_mean + region_mean):
            raise RuntimeError(f"loss is not finite at epoch {epoch}, last.pt keeps the previous epoch")

        scheduler.step()
        if not main_process:
            continue
        elapsed = time.time() - started
        memory = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        steps = totals[2].item() / world
        print(
            f"epoch {epoch} lr {lr:.2e} ddpm {ddpm_mean:.4f} region {region_mean:.4f} "
            f"steps/s {steps / elapsed:.2f} epoch_min {elapsed / 60:.1f} gpu_gb {memory:.1f}",
            flush=True,
        )

        state = {
            "rddm": rddm.state_dict(),
            "cond_net1": cond_net1.state_dict(),
            "cond_net2": cond_net2.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        }
        # Write then rename: a job killed mid-save won't corrupt last.pt
        torch.save(state, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)

        if epoch % config["save_every"] == 0 or epoch == config["n_epoch"] - 1:
            torch.save(rddm.state_dict(), f"{path}/RDDM_epoch{epoch}.pth")
            torch.save(cond_net1.state_dict(), f"{path}/ConditionNet1_epoch{epoch}.pth")
            torch.save(cond_net2.state_dict(), f"{path}/ConditionNet2_epoch{epoch}.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(ROOT))
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--path", required=True, help="checkpoint directory")
    parser.add_argument("--n-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128, help="total over all GPUs")
    parser.add_argument("--micro-batch", type=int, default=128, help="largest batch one GPU fits at once")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None, help="limit steps per epoch for a speed test")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    # torchrun sets these; a plain python run is one process
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    elif torch.backends.mps.is_available() and world == 1:
        device = "mps"
    else:
        device = "cpu"
    if "SLURM_JOB_ID" in os.environ and not device.startswith("cuda"):
        raise RuntimeError("no GPU available in a Slurm job, refusing to train on CPU")
    if world > 1:
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")

    # 512 over 3 GPUs is 171 per GPU, 513 in total
    per_rank_batch = math.ceil(args.batch_size / world)
    if rank == 0:
        print(f"device {device}, world {world}, batch per GPU {per_rank_batch}, {vars(args)}", flush=True)

    dataset = load_split(args.data_root, "train", args.datasets)
    if rank == 0:
        print(f"train windows: {len(dataset)}", flush=True)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) // world - 1)
    dataloader = DataLoader(
        dataset, batch_size=per_rank_batch, shuffle=sampler is None, sampler=sampler, drop_last=True,
        num_workers=workers, pin_memory=device.startswith("cuda"), persistent_workers=workers > 0,
    )

    config = {
        "n_epoch": args.n_epoch,
        "t_warmup": 20,
        "lr": 1e-4,
        "nT": 10,
        "attention_heads": 8,
        "alpha1": 100,
        "alpha2": 1,
        "save_every": args.save_every,
        "max_steps": args.max_steps,
        "micro_batch": args.micro_batch,
        "seed": args.seed,
        "device": device,
        "path": args.path,
        "rank": rank,
        "local_rank": local_rank,
        "world": world,
    }
    train_rddm(config, dataloader, sampler)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
