"""Phase 6: train the CQT-LCNN.

Usage:
    python -m src.train_lcnn --tag baseline
    python -m src.train_lcnn --tag T150 --n-frames 150
    python -m src.train_lcnn --tag cmvn --norm cmvn
    python -m src.train_lcnn --tag flatten --head flatten

Each run writes to results/phase6/<tag>/ and models/lcnn_<tag>.pt, and is resumable
(re-running picks up from the last epoch checkpoint).

Two design points worth knowing:

*   The LR scheduler and checkpointing key on **dev EER, not dev loss**. Under the
    3.7:1 imbalance the loss is dominated by the majority class, so it can fall
    while the bonafide-vs-spoof RANKING -- which is what EER measures and what we
    report -- gets worse.

*   Train EER is measured on a held-in subsample with augmentation OFF and a
    centre crop, i.e. scored exactly like dev. Measuring it under augmentation
    would make the train/dev gap meaningless. That gap is the planned diagnostic
    for whether waveform-domain augmentation is needed (PROJECT_PLAN.md phase 6):
    if dev EER plateaus while train EER keeps falling, the model is memorising and
    more augmentation variety would genuinely pay.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from . import config, metrics
from .datasets import CQTDataset
from .models_lcnn import LCNN

TRAIN_EVAL_N = 20_000   # subsample size for the train-EER diagnostic


def build_loaders(args):
    train_ds = CQTDataset("train", train=True, n_frames=args.n_frames, norm=args.norm,
                          wav_aug_copies=args.wav_aug_copies, p_clean=args.p_clean)
    dev_ds = CQTDataset("dev", train=False, n_frames=args.n_frames, norm=args.norm)

    # Same underlying data as train_ds but scored like dev: no augmentation, centre
    # crop. Subsampled purely for speed.
    train_eval_full = CQTDataset("train", train=False, n_frames=args.n_frames,
                                 norm=args.norm, augment=False)
    rng = np.random.default_rng(config.RANDOM_SEED)
    pick = rng.choice(len(train_eval_full), min(TRAIN_EVAL_N, len(train_eval_full)),
                      replace=False)
    train_eval_ds = Subset(train_eval_full, pick.tolist())

    common = dict(num_workers=args.num_workers, pin_memory=True,
                  persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **common)
    dev_dl = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    train_eval_dl = DataLoader(train_eval_ds, batch_size=args.eval_batch_size,
                               shuffle=False, **common)
    return train_ds, train_dl, dev_dl, train_eval_dl, dev_ds


@torch.no_grad()
def evaluate(model, loader, device, use_amp, desc) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, labels = [], []
    for x, y in tqdm(loader, desc=desc, leave=False, unit="batch"):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
        scores.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def train_one_epoch(model, loader, opt, scaler, criterion, device, use_amp, epoch):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(loader, desc=f"epoch {epoch}", leave=False, unit="batch")
    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(x), y)
        # GradScaler is required with AMP: FP16 gradients can underflow to zero, so
        # the loss is scaled up before backward and unscaled before the step.
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
        bar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(n, 1)


def plot_curves(log: pd.DataFrame, run_dir: Path, tag: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(log.epoch, log.train_loss, marker="o")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=.3)

    axes[1].plot(log.epoch, log.train_eer * 100, marker="o", label="train (no aug, centre crop)")
    axes[1].plot(log.epoch, log.dev_eer * 100, marker="s", label="dev")
    best = log.loc[log.dev_eer.idxmin()]
    axes[1].axvline(best.epoch, ls="--", c="tab:red", alpha=.6,
                    label=f"best dev {best.dev_eer*100:.3f}%")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("EER (%)")
    axes[1].set_title("EER — the train/dev gap is the\naugmentation diagnostic")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

    axes[2].plot(log.epoch, (log.train_eer - log.dev_eer) * 100, marker="o", c="tab:purple")
    axes[2].axhline(0, c="k", lw=.8)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("train EER − dev EER (pp)")
    axes[2].set_title("Generalisation gap\n(diverging ⇒ memorising)")
    axes[2].grid(alpha=.3)

    fig.suptitle(f"CQT-LCNN — {tag}")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=150)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="run name; outputs go to results/phase6/<tag>/")
    p.add_argument("--n-frames", type=int, default=config.CQT_FIXED_FRAMES)
    p.add_argument("--norm", choices=["unit", "cmvn"], default=config.LCNN_NORM)
    p.add_argument("--head", choices=["timepool", "flatten"], default=config.LCNN_HEAD)
    p.add_argument("--epochs", type=int, default=config.LCNN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.LCNN_BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=config.LCNN_BATCH_SIZE * 2)
    p.add_argument("--lr", type=float, default=config.LCNN_LR)
    p.add_argument("--num-workers", type=int, default=config.LCNN_NUM_WORKERS)
    p.add_argument("--wav-aug-copies", type=int, default=0,
                   help="number of pre-computed waveform-augmented copies to sample "
                        "from (0 = clean only). Requires src.augment_waveform first.")
    p.add_argument("--p-clean", type=float, default=None,
                   help="probability of drawing the CLEAN blob instead of an augmented "
                        "copy (PROJECT_PLAN 9.1.1). Default None = uniform over "
                        "{clean, aug1..N}, i.e. p(clean)=1/(N+1), the Phase 6 behaviour. "
                        "Setting this decouples dose from diversity without generating "
                        "any new copies.")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore existing checkpoint")
    args = p.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    run_dir = config.PHASE6_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.PHASE6_MODELS_DIR / f"lcnn_{args.tag}.pt"
    best_ckpt_path = config.PHASE6_MODELS_DIR / f"lcnn_{args.tag}_best.pt"
    log_path = run_dir / "log.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.LCNN_USE_AMP and not args.no_amp and device.type == "cuda"

    print(f"run '{args.tag}': T={args.n_frames}, norm={args.norm}, head={args.head}, "
          f"batch={args.batch_size}, amp={use_amp}, wav_aug={args.wav_aug_copies}, "
          f"device={device}")

    train_ds, train_dl, dev_dl, train_eval_dl, dev_ds = build_loaders(args)
    pos_weight = train_ds.pos_weight()
    print(f"  train {len(train_ds):,}  dev {len(dev_ds):,}  pos_weight={pos_weight:.3f} "
          f"(n_spoof/n_bonafide)")

    model = LCNN(n_frames=args.n_frames, head=args.head).to(device)
    print(f"  model: {model.n_params():,} params, fc_in={model.fc_in}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=config.LCNN_LR_FACTOR, patience=config.LCNN_LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_eer, rows = 1, float("inf"), []
    if ckpt_path.exists() and not args.force:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_eer = ck["best_eer"]
        rows = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
        print(f"  resumed from epoch {ck['epoch']} (best dev EER {best_eer*100:.3f}%)")

    epochs_since_best = 0
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, opt, scaler, criterion, device, use_amp, epoch)

        dev_scores, dev_labels = evaluate(model, dev_dl, device, use_amp, "dev")
        dev_eer, dev_thr = metrics.eer_from_labels(dev_labels, dev_scores)
        tr_scores, tr_labels = evaluate(model, train_eval_dl, device, use_amp, "train-eval")
        train_eer, _ = metrics.eer_from_labels(tr_labels, tr_scores)

        sched.step(dev_eer)
        lr_now = opt.param_groups[0]["lr"]
        secs = time.time() - t0
        rows.append({"epoch": epoch, "train_loss": loss, "train_eer": train_eer,
                     "dev_eer": dev_eer, "dev_threshold": dev_thr, "lr": lr_now,
                     "seconds": secs})
        pd.DataFrame(rows).to_csv(log_path, index=False)

        improved = dev_eer < best_eer
        marker = "  <-- best" if improved else ""
        print(f"  epoch {epoch:>3}: loss={loss:.4f}  train EER={train_eer*100:6.3f}%  "
              f"dev EER={dev_eer*100:6.3f}%  lr={lr_now:.2e}  {secs/60:.1f}min{marker}")

        if improved:
            best_eer, epochs_since_best = dev_eer, 0
            pd.DataFrame({"filename": dev_ds.index["filename"].to_numpy(),
                          "score": dev_scores,
                          "label": np.where(dev_labels == 1, "bonafide", "spoof")}
                         ).to_csv(run_dir / "dev_scores.csv", index=False)
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_eer": best_eer,
                        "args": vars(args)}, best_ckpt_path)
        else:
            epochs_since_best += 1

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_eer": best_eer}, ckpt_path)

        if epochs_since_best >= config.LCNN_EARLY_STOP_PATIENCE:
            print(f"  early stop: no dev-EER improvement for "
                  f"{config.LCNN_EARLY_STOP_PATIENCE} epochs")
            break

    log = pd.DataFrame(rows)
    plot_curves(log, run_dir, args.tag)

    best_row = log.loc[log.dev_eer.idxmin()]
    scores = pd.read_csv(run_dir / "dev_scores.csv")
    report = metrics.full_report((scores.label == "bonafide").astype(int).to_numpy(),
                                 scores.score.to_numpy())
    summary = {"tag": args.tag, "n_frames": args.n_frames, "norm": args.norm,
               "head": args.head, "params": model.n_params(),
               "best_epoch": int(best_row.epoch), "epochs_run": int(log.epoch.max()),
               "train_eer_at_best": float(best_row.train_eer), **report}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    gap = (best_row.train_eer - best_row.dev_eer) * 100
    print(f"\nbest dev EER {best_row.dev_eer*100:.3f}% at epoch {int(best_row.epoch)}  "
          f"(train EER {best_row.train_eer*100:.3f}%, gap {gap:+.2f} pp)")
    print(f"written to {run_dir}")


if __name__ == "__main__":
    main()
