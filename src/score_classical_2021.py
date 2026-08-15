"""Phase 7, pass 2: score the Phase 5 classical baselines on 2021.

    python -m src.score_classical_2021              # ~1h, resumable
    python -m src.score_classical_2021 --models MFCC-SVM

Reads the pooled MFCC that pass 1 cached, so no audio is decoded here and nothing
is extracted twice. This runs as a SEPARATE pass rather than inside
evaluate_2021.py on purpose: libsvm scoring is single-threaded CPU work
(~3.56 ms/file, measured in Phase 5 at 37,943 support vectors) and would otherwise
contend with the 8 extraction worker processes for the same cores, slowing the
expensive pass to speed up the cheap one.

Both models were fixed in Phase 5 and are pre-registered in
config.PHASE7_CLASSICAL_SYSTEMS. Score orientation matches the project convention
in metrics.py -- higher = more bonafide -- for both: SVC.decision_function is
signed toward the positive class (1 = bonafide), and predict_proba column 1 is
P(bonafide).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .train_classical import RF_FULL_MODEL, SVM_MODEL, score_in_chunks

MFCC_COLS = [f"mfcc_{i}" for i in range(120)]

# tag -> (path, how to turn the fitted model into a bonafide-oriented score)
MODELS = {
    "MFCC-SVM": (SVM_MODEL, "decision"),
    "MFCC-RF": (RF_FULL_MODEL, "proba"),
}


def load_model(tag: str):
    path, kind = MODELS[tag]
    if not path.exists():
        raise FileNotFoundError(f"missing Phase 5 model for {tag}: {path}")
    model = joblib.load(path)
    if kind == "proba":
        # 483MB, 300 trees. n_jobs is baked into the pickle; reset it so scoring
        # uses the cores this machine actually has free.
        try:
            model.n_jobs = config.RF_N_JOBS
        except AttributeError:
            pass
    return model, kind


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=list(config.PHASE7_CLASSICAL_SYSTEMS))
    p.add_argument("--force", action="store_true", help="rescore shards already done")
    args = p.parse_args()

    shards = sorted(config.PA2021_MFCC_SHARD_DIR.glob("mfcc_*.parquet"))
    if not shards:
        raise RuntimeError("no cached 2021 MFCC -- run src.evaluate_2021 first")
    print(f"{len(shards)} MFCC shards; models: {', '.join(args.models)}")

    for tag in args.models:
        model, kind = load_model(tag)
        out_dir = config.PA2021_CLASSICAL_SHARD_DIR / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        todo = [s for s in shards if args.force or not (out_dir / s.name).exists()]
        print(f"\n{tag}: {len(shards) - len(todo)}/{len(shards)} shards already scored")

        t0, n = time.time(), 0
        for s in tqdm(todo, desc=tag, unit="shard"):
            df = pd.read_parquet(s)
            scores = score_in_chunks(model, df[MFCC_COLS].to_numpy(), kind,
                                     desc=f"{tag} {s.stem}")
            res = pd.DataFrame({"filename": df["filename"], tag: scores.astype(np.float32)})
            tmp = out_dir / (s.name + ".tmp")
            res.to_parquet(tmp, index=False)
            os.replace(tmp, out_dir / s.name)
            n += len(df)
        if n:
            print(f"  {n:,} files in {(time.time()-t0)/3600:.2f}h "
                  f"({(time.time()-t0)/n*1000:.2f} ms/file)")
        del model

    # Merge every model's shards into one table, joined on filename.
    merged = None
    for tag in args.models:
        out_dir = config.PA2021_CLASSICAL_SHARD_DIR / tag
        files = sorted(out_dir.glob("mfcc_*.parquet"))
        if not files:
            continue
        part = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        merged = part if merged is None else merged.merge(part, on="filename", how="outer")

    if merged is not None:
        tmp = config.PA2021_CLASSICAL_SCORES.with_name(
            config.PA2021_CLASSICAL_SCORES.name + ".tmp")
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, config.PA2021_CLASSICAL_SCORES)
        print(f"\nmerged -> {config.PA2021_CLASSICAL_SCORES} ({len(merged):,} rows, "
              f"columns {[c for c in merged.columns if c != 'filename']})")


if __name__ == "__main__":
    main()
