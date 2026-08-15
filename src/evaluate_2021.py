"""Phase 7, pass 1: stream the held-out 2021 PA eval set and score every LCNN.

    python -m src.evaluate_2021                 # the real run (~4h, resumable)
    python -m src.evaluate_2021 --limit 2000    # smoke test
    python -m src.evaluate_2021 --verify-only   # byte-check the cache, no scoring
    python -m src.evaluate_2021 --merge-only    # rebuild merged index/scores

THIS IS THE ONE PASS OVER THE HELD-OUT SET. The systems it scores were fixed in
config.PHASE7_LCNN_SYSTEMS before any 2021 number existed (PROJECT_PLAN.md phase 7);
nothing here selects, tunes or thresholds against 2021.

Structure, and why:

*   ONE extraction, many models. Extraction is the expensive half (~55.5 ms/file
    measured warm: 28.7 decode + 3.6 MFCC + 23.2 CQT) and scoring is cheap
    (~0.95 ms/sample/model), so every model is scored inside the same pass rather
    than re-reading the corpus once per system.

*   The 7 LCNNs need only FOUR distinct inputs -- (T400, unit), (T400, cmvn),
    (T250, unit), (T150, unit) -- so each chunk's CQT is windowed four times, not
    seven, and each windowed batch is shown to every model that wants it. Four of
    the seven share the (T400, unit) view.

*   Windowing and normalisation come from datasets.py, the exact functions the
    Phase 6 dev evaluation used. A divergence there would silently invalidate the
    dev-vs-eval comparison this whole phase exists to make.

*   CQT is cached (~12.7GB) as insurance, not for speed: the dominant risk is
    finding a bug AFTER a 4-hour pass, and cached, a re-score costs ~1.5h of GPU
    and no CPU. See config.PHASE7_CACHE_CQT for the sizing.

*   Resumable at chunk granularity. The LCNN score shard is written LAST, so its
    existence marks a chunk complete; every write is a temp-file rename, so a crash
    mid-write cannot leave a truncated shard that resume would trust.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from . import config
from .datasets import fit_length, normalise
from .features import extract_cqt_uint8, extract_for_eval, load_audio
from .models_lcnn import LCNN

MFCC_COLS = [f"mfcc_{i}" for i in range(120)]
FAILURES_DIR = config.PA2021_WORK_DIR / "failures"
FAILURES_DIR.mkdir(parents=True, exist_ok=True)


# --- extraction ----------------------------------------------------------------
# The worker is features.extract_for_eval, deliberately defined in a torch-free
# module -- see its docstring. Importing it here is safe; loky pickles it by
# reference, so workers import src.features and not this module.

def extract_chunk(records: list[dict], n_jobs: int) -> list[dict]:
    return Parallel(n_jobs=n_jobs)(delayed(extract_for_eval)(r) for r in records)


# --- models --------------------------------------------------------------------

def load_systems(tags: list[str], device) -> dict:
    """Rebuild each pre-registered LCNN from its own checkpoint.

    Architecture comes from the `args` dict saved alongside the weights, never from
    config or from the tag string: a model silently rebuilt at the wrong T or head
    would still load and still produce plausible scores.
    """
    systems = {}
    for tag in tags:
        path = config.PHASE6_MODELS_DIR / f"lcnn_{tag}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing pre-registered checkpoint: {path}")
        ck = torch.load(path, map_location=device, weights_only=False)
        a = ck["args"]
        model = LCNN(n_frames=a["n_frames"], head=a["head"]).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        systems[tag] = {"model": model, "T": a["n_frames"], "norm": a["norm"],
                        "head": a["head"], "dev_eer": float(ck["best_eer"]),
                        "epoch": int(ck["epoch"])}
    return systems


def group_views(systems: dict) -> dict:
    """Map (T, norm) -> [tags], so each distinct input is built once per chunk."""
    views: dict[tuple[int, str], list[str]] = {}
    for tag, s in systems.items():
        views.setdefault((s["T"], s["norm"]), []).append(tag)
    return views


@torch.no_grad()
def score_chunk(systems: dict, views: dict, cqts: list[np.ndarray],
                device, batch: int) -> dict[str, np.ndarray]:
    """Window each chunk once per view and run every model that shares that view.

    Batched rather than done whole-chunk because a 4,000-file chunk at T=400 in
    float32 would be 576MB per view on a 5.9GB machine; at batch 64 it is 9.2MB.
    """
    out: dict[str, np.ndarray] = {}
    for (T, norm), tags in views.items():
        acc = {t: [] for t in tags}
        for i in range(0, len(cqts), batch):
            arr = np.stack([normalise(fit_length(c, T), norm) for c in cqts[i:i + batch]])
            x = torch.from_numpy(arr).unsqueeze(1).to(device, non_blocking=True)
            for t in tags:
                acc[t].append(systems[t]["model"](x).float().cpu().numpy())
        for t in tags:
            out[t] = np.concatenate(acc[t])
    return out


# --- atomic shard writes -------------------------------------------------------

def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_blob_shard(path: Path, arrays: list[np.ndarray]) -> list[int]:
    """Concatenate this chunk's CQTs into one file, returning each one's offset.

    Variable-length records, so the offsets are data and must be stored -- position
    cannot be recomputed arithmetically (same constraint as Phase 6's packer).
    """
    tmp = path.with_name(path.name + ".tmp")
    offsets, off = [], 0
    with open(tmp, "wb") as fh:
        for a in arrays:
            fh.write(np.ascontiguousarray(a).tobytes())
            offsets.append(off)
            off += a.nbytes
    os.replace(tmp, path)
    return offsets


def shard_paths(chunk: int) -> tuple[Path, Path, Path, Path]:
    return (config.PA2021_CQT_SHARD_DIR / f"cqt_{chunk:05d}.dat",
            config.PA2021_CQT_SHARD_DIR / f"index_{chunk:05d}.parquet",
            config.PA2021_MFCC_SHARD_DIR / f"mfcc_{chunk:05d}.parquet",
            config.PA2021_SCORE_SHARD_DIR / f"scores_{chunk:05d}.parquet")


def chunk_complete(chunk: int, expected: int) -> bool:
    """A chunk counts as done only when it holds a score for EVERY file in it.

    Not merely "the shard exists". The first smoke run wrote short shards after
    transient memory failures and reported success while silently dropping 11% of
    the set -- exactly the kind of hole that would corrupt a headline EER with no
    error anywhere. Short shards are therefore redone on the next run, since the
    failures that produce them (commit-limit pressure) are transient. A file that
    genuinely cannot decode would otherwise loop forever, so `--accept-failures`
    writes a flag that pins the chunk as permanently accepted once understood.
    """
    if (config.PA2021_SCORE_SHARD_DIR / f"accepted_{chunk:05d}.flag").exists():
        return True
    score_p = shard_paths(chunk)[3]
    if not score_p.exists():
        return False
    return len(pd.read_parquet(score_p, columns=["filename"])) == expected


def write_chunk(chunk: int, ok: list[dict], meta: pd.DataFrame,
                scores: dict[str, np.ndarray], cache_cqt: bool) -> None:
    """Order matters: the score shard is written LAST and is the completion marker,
    so a crash between writes leaves the chunk incomplete and it is simply redone
    (extraction is deterministic, so redoing it reproduces identical bytes)."""
    blob_p, index_p, mfcc_p, score_p = shard_paths(chunk)
    names = [r["filename"] for r in ok]
    m = meta.set_index("filename").loc[names]

    if cache_cqt:
        offsets = _write_blob_shard(blob_p, [r["cqt"] for r in ok])
        _atomic_parquet(pd.DataFrame({
            "filename": names, "shard": chunk, "offset": offsets,
            "n_frames": [r["cqt"].shape[1] for r in ok],
            "label": m["label"].to_numpy(), "partition": m["partition"].to_numpy(),
        }), index_p)

    mfcc_df = pd.DataFrame(np.stack([r["mfcc"] for r in ok]), columns=MFCC_COLS)
    mfcc_df.insert(0, "filename", names)
    _atomic_parquet(mfcc_df, mfcc_p)

    out = pd.DataFrame({"filename": names})
    out["n_frames"] = [r["cqt"].shape[1] for r in ok]
    out["duration_s"] = [r["n_samples"] / config.SAMPLE_RATE for r in ok]
    for tag, v in scores.items():
        out[tag] = v.astype(np.float32)
    _atomic_parquet(out, score_p)


# --- merge / verify ------------------------------------------------------------

def merge_shards(quiet: bool = False) -> pd.DataFrame:
    score_files = sorted(config.PA2021_SCORE_SHARD_DIR.glob("scores_*.parquet"))
    if not score_files:
        raise RuntimeError("no score shards to merge -- run the extraction pass first")
    scores = pd.concat([pd.read_parquet(p) for p in score_files], ignore_index=True)
    _atomic_parquet(scores, config.PA2021_LCNN_SCORES)

    index_files = sorted(config.PA2021_CQT_SHARD_DIR.glob("index_*.parquet"))
    if index_files:
        idx = pd.concat([pd.read_parquet(p) for p in index_files], ignore_index=True)
        _atomic_parquet(idx, config.PA2021_CQT_INDEX)

    fail_files = sorted(FAILURES_DIR.glob("*.parquet"))
    if fail_files:
        fails = pd.concat([pd.read_parquet(p) for p in fail_files], ignore_index=True)
        fails.to_csv(config.PA2021_FAILURES_CSV, index=False)

    if not quiet:
        print(f"merged {len(score_files)} score shards -> {len(scores):,} rows "
              f"at {config.PA2021_LCNN_SCORES}")
        if index_files:
            print(f"merged {len(index_files)} index shards -> {config.PA2021_CQT_INDEX}")
        if fail_files:
            print(f"{len(fails):,} extraction failures -> {config.PA2021_FAILURES_CSV}")
    return scores


def read_cached_cqt(row) -> np.ndarray:
    blob = config.PA2021_CQT_SHARD_DIR / f"cqt_{int(row['shard']):05d}.dat"
    with open(blob, "rb", buffering=0) as fh:
        fh.seek(int(row["offset"]))
        raw = fh.read(config.CQT_N_BINS * int(row["n_frames"]))
    return np.frombuffer(raw, dtype=np.uint8).reshape(config.CQT_N_BINS, int(row["n_frames"]))


def verify_cache(n: int, manifest: pd.DataFrame) -> bool:
    """Re-extract random files from source and compare BYTE-FOR-BYTE to the cache.

    Same check pack_features.py runs, for the same reason: a wrong offset would
    corrupt every score downstream while surfacing no error anywhere.
    """
    if not config.PA2021_CQT_INDEX.exists():
        print("no merged CQT index -- nothing to verify"); return False
    idx = pd.read_parquet(config.PA2021_CQT_INDEX)
    paths = manifest.set_index("filename")["filepath"]
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = idx.iloc[rng.choice(len(idx), min(n, len(idx)), replace=False)]

    bad = 0
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="verify", unit="file"):
        fresh = extract_cqt_uint8(load_audio(paths[row["filename"]]))
        if not np.array_equal(fresh, read_cached_cqt(row)):
            bad += 1
            print(f"  MISMATCH: {row['filename']}")
    print(f"verify: {len(sample) - bad}/{len(sample)} byte-identical to a fresh extraction")
    return bad == 0


# --- main ----------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, default=config.PHASE7_CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=config.PHASE7_EVAL_BATCH)
    p.add_argument("--n-jobs", type=int, default=config.FEATURE_EXTRACTION_N_JOBS)
    p.add_argument("--limit", type=int, default=0, help="only the first N files (smoke test)")
    p.add_argument("--no-prefetch", action="store_true",
                   help="do not extract the next chunk while the GPU scores this one")
    p.add_argument("--no-cache-cqt", action="store_true")
    p.add_argument("--max-failure-rate", type=float, default=0.01,
                   help="abort if this fraction of files fails to extract")
    p.add_argument("--accept-failures", action="store_true",
                   help="pin short chunks as permanently complete (use only after "
                        "confirming the failures are genuine undecodable files)")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--verify-n", type=int, default=config.PHASE7_VERIFY_N)
    args = p.parse_args()

    # Pin the workers to one BLAS/FFT thread each. librosa's FFT threads internally,
    # so 8 worker processes each spawning threads oversubscribe the 12 logical cores.
    # Measured: 36 -> 40 files/s. Set before joblib spawns, since loky passes the
    # parent environment to its children, which import numpy fresh.
    # (n_jobs itself is already saturated -- 12 workers measured no faster than 8.)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    manifest = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet")
    files = manifest[manifest["partition"].isin(config.PHASE7_PARTITIONS)]
    # Sort by filename so chunk boundaries -- and therefore shard contents -- are
    # identical on every run, which is what makes resume-by-shard-existence sound.
    files = files.sort_values("filename").reset_index(drop=True)
    if args.limit:
        files = files.iloc[:args.limit].reset_index(drop=True)

    if args.merge_only:
        merge_shards(); return
    if args.verify_only:
        sys.exit(0 if verify_cache(args.verify_n, files) else 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    systems = load_systems(list(config.PHASE7_LCNN_SYSTEMS), device)
    views = group_views(systems)
    cache_cqt = config.PHASE7_CACHE_CQT and not args.no_cache_cqt

    print(f"Phase 7 -- scoring the HELD-OUT 2021 PA set. Device: {device}")
    print(f"  partitions {list(config.PHASE7_PARTITIONS)}: {len(files):,} files")
    print(f"  {len(systems)} pre-registered LCNNs in {len(views)} distinct input views:")
    for (T, norm), tags in sorted(views.items()):
        print(f"    T={T:<4} norm={norm:<5} -> {', '.join(tags)}")
    print(f"  cache CQT: {cache_cqt}   prefetch: {not args.no_prefetch}   "
          f"chunk {args.chunk_size}   batch {args.batch_size}")

    starts = list(range(0, len(files), args.chunk_size))
    sizes = [min(args.chunk_size, len(files) - s) for s in starts]
    todo = [c for c in range(len(starts)) if not chunk_complete(c, sizes[c])]
    done = len(starts) - len(todo)
    if done:
        print(f"  resuming: {done}/{len(starts)} chunks already complete")
    if not todo:
        print("  nothing to do."); merge_shards(); return

    def records_for(c: int) -> list[dict]:
        return files.iloc[starts[c]:starts[c] + args.chunk_size].to_dict("records")

    t_start, n_done, n_failed = time.time(), 0, 0
    bar = tqdm(todo, desc="chunks", unit="chunk")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None if args.no_prefetch else pool.submit(
            extract_chunk, records_for(todo[0]), args.n_jobs)

        for i, c in enumerate(bar):
            results = (pending.result() if pending is not None
                       else extract_chunk(records_for(c), args.n_jobs))
            # Kick off the next chunk's extraction (8 CPU procs) before doing this
            # chunk's GPU work, so the two overlap instead of alternating.
            pending = (pool.submit(extract_chunk, records_for(todo[i + 1]), args.n_jobs)
                       if not args.no_prefetch and i + 1 < len(todo) else None)

            ok = [r for r in results if r["error"] is None]
            bad = [r for r in results if r["error"] is not None]
            if bad:
                n_failed += len(bad)
                _atomic_parquet(pd.DataFrame([{"filename": r["filename"],
                                               "error": r["error"]} for r in bad]),
                                FAILURES_DIR / f"fail_{c:05d}.parquet")
            else:
                (FAILURES_DIR / f"fail_{c:05d}.parquet").unlink(missing_ok=True)
            if ok:
                scores = score_chunk(systems, views, [r["cqt"] for r in ok],
                                     device, args.batch_size)
                write_chunk(c, ok, files, scores, cache_cqt)
            if bad and args.accept_failures:
                (config.PA2021_SCORE_SHARD_DIR / f"accepted_{c:05d}.flag").write_text(
                    f"{len(bad)} permanently-failed files accepted\n", encoding="utf-8")
            n_done += len(results)
            # Free the chunk's arrays before the next one lands. With prefetch on,
            # the next chunk is already being built in worker processes while this
            # runs, so two chunks' worth can otherwise be resident at once.
            del results, ok, bad
            # Abort loudly on a systemic failure rather than grinding through the
            # whole corpus dropping files. Phase 4 decoded 241,056 files with ZERO
            # failures via this same ffmpeg path, so any sustained rate here means
            # something environmental (commit limit, disk) that re-running will not
            # fix and that must not be silently absorbed into a headline number.
            if n_failed > max(5, args.max_failure_rate * n_done):
                raise RuntimeError(
                    f"{n_failed}/{n_done} files failed to extract "
                    f"({n_failed / n_done:.1%}), above --max-failure-rate "
                    f"{args.max_failure_rate:.1%}. Inspect "
                    f"{FAILURES_DIR}; completed chunks are kept, so fixing the "
                    f"cause and re-running resumes from here.")
            rate = n_done / max(time.time() - t_start, 1e-9)
            remaining = sum(sizes[x] for x in todo[i + 1:])
            bar.set_postfix(files=f"{n_done:,}", rate=f"{rate:.0f}/s",
                            eta=f"{remaining / max(rate, 1e-9) / 3600:.1f}h",
                            fail=n_failed)

    elapsed = (time.time() - t_start) / 3600
    print(f"\nextraction+scoring done: {n_done:,} files in {elapsed:.2f}h "
          f"({n_done / max(elapsed * 3600, 1e-9):.0f} files/s), {n_failed} failures")

    scores = merge_shards()
    missing = set(files["filename"]) - set(scores["filename"])
    if missing:
        print(f"WARNING: {len(missing):,} files have no score (see "
              f"{config.PA2021_FAILURES_CSV}); every reported metric is over the rest")

    if cache_cqt and args.verify_n:
        verify_cache(args.verify_n, files)


if __name__ == "__main__":
    main()
