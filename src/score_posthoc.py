"""Score the POST-HOC systems on 2021, from the cached CQT.

    python -m src.score_posthoc          # ~12 min, one pass over the cache
    python -m src.score_posthoc --force  # rescore even if the output exists

These systems (config.PHASE7_POSTHOC_SYSTEMS) were trained AFTER 2021 results were
seen, so they carry none of the pre-registration guarantee the Phase 7 table does.
Everything they produce is written to its own files and reported in its own section.

THE DECISION RULE IS ENFORCED HERE, IN CODE. PROJECT_PLAN.md 9.3.1 declared, before
these runs were scored, that both candidates would be compared on `progress` and that
only the winner would ever be scored on `eval`. That is expressed as the per-system
partition whitelist in config, and rows outside it are filtered out BEFORE the forward
pass -- so the loser's eval scores are never computed at all, and no "best of two on
eval" can be reported even by mistake.

Reads from the Phase 7 CQT cache, so no audio is decoded and nothing is re-extracted.
Chunked one shard at a time: 721,332 eval CQTs held at once would need ~9.7GB, which
this machine does not have.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=config.PHASE7_EVAL_BATCH)
    args = p.parse_args()

    import torch
    from .evaluate_2021 import shard_paths, score_chunk
    from .models_lcnn import LCNN

    if config.PA2021_POSTHOC_SCORES.exists() and not args.force:
        print(f"{config.PA2021_POSTHOC_SCORES} exists; --force to rescore")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    systems, allowed, pending = {}, {}, []
    for tag, parts in config.PHASE7_POSTHOC_SYSTEMS.items():
        path = config.PHASE6_MODELS_DIR / f"lcnn_{tag}_best.pt"
        if not path.exists():
            # SKIP, not raise. The registry is populated when a sweep is DECLARED
            # (9.1.1 registers all five dose points up front so none can reach eval by
            # accident), which is necessarily before the runs finish. Aborting the pass
            # would mean an incremental sweep could never be scored until every run was
            # done. Skipping is announced loudly and listed again at the end, so a
            # genuinely forgotten checkpoint is still impossible to miss.
            pending.append(tag)
            print(f"  {tag}: no checkpoint yet -- SKIPPED (not trained)")
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        a = ck["args"]
        m = LCNN(n_frames=a["n_frames"], head=a["head"]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        systems[tag] = {"model": m, "T": a["n_frames"], "norm": a["norm"]}
        allowed[tag] = set(parts)
        print(f"{tag}: T={a['n_frames']} head={a['head']} epoch={ck['epoch']} "
              f"dev_eer={ck['best_eer']*100:.3f}%  partitions={sorted(parts)}")

    frames, t0 = [], time.time()
    for c in range(236):
        blob_p, index_p, _, _ = shard_paths(c)
        if not index_p.exists():
            continue
        idx = pd.read_parquet(index_p)

        # Read this shard's CQTs once, then score each system on ONLY the rows its
        # whitelist permits. Filtering before the forward pass is what makes the
        # rule structural rather than advisory.
        need = idx[idx["partition"].isin(set().union(*allowed.values()))]
        if not len(need):
            continue
        arrs, pos = {}, {}
        with open(blob_p, "rb", buffering=0) as fh:
            for i, (off, nf) in enumerate(zip(need["offset"].to_numpy(),
                                              need["n_frames"].to_numpy())):
                fh.seek(int(off))
                raw = fh.read(config.CQT_N_BINS * int(nf))
                arrs[i] = np.frombuffer(raw, dtype=np.uint8).reshape(
                    config.CQT_N_BINS, int(nf))
        parts_col = need["partition"].to_numpy()
        out = pd.DataFrame({"filename": need["filename"].to_numpy(),
                            "partition": parts_col})
        for tag, s in systems.items():
            keep = np.flatnonzero(np.isin(parts_col, list(allowed[tag])))
            col = np.full(len(need), np.nan, dtype=np.float32)
            if len(keep):
                sub = [arrs[i] for i in keep]
                col[keep] = score_chunk({tag: s}, {(s["T"], s["norm"]): [tag]},
                                        sub, device, args.batch_size)[tag]
            out[tag] = col
        frames.append(out)
        del arrs
        if c % 40 == 0:
            done = sum(len(f) for f in frames)
            print(f"  shard {c:3d}/236  {done:,} rows  {(time.time()-t0)/60:.1f} min")

    d = pd.concat(frames, ignore_index=True)
    tmp = config.PA2021_POSTHOC_SCORES.with_suffix(".parquet.tmp")
    d.to_parquet(tmp, index=False)
    os.replace(tmp, config.PA2021_POSTHOC_SCORES)
    print(f"\nscored in {(time.time()-t0)/60:.1f} min -> {config.PA2021_POSTHOC_SCORES}")
    for tag in systems:
        n = d.groupby("partition")[tag].apply(lambda v: int(v.notna().sum()))
        print(f"  {tag:20s} " + "  ".join(f"{k}={v:,}" for k, v in n.items() if v))

    # score.txt exports, same official format as everything else
    for tag in systems:
        sub = d.loc[d[tag].notna(), ["filename", tag]]
        sub.to_csv(config.PHASE7_POSTHOC_SCORES_DIR / f"{tag}.score.txt",
                   sep=" ", header=False, index=False, float_format="%.6f")
    print(f"  exported {len(systems)} score.txt -> {config.PHASE7_POSTHOC_SCORES_DIR}")
    if pending:
        print(f"\n  NOT SCORED, no checkpoint yet: {', '.join(pending)}")
        print("  Re-run this after training them; scoring is idempotent.")


if __name__ == "__main__":
    main()
