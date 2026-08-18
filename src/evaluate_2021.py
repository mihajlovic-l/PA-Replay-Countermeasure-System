"""Phase 7, pass 1: extract the held-out 2021 PA set, then score every LCNN.

    python -m src.evaluate_2021                    # extract, then score (~6.5h)
    python -m src.evaluate_2021 --stage extract    # CPU only, no torch loaded
    python -m src.evaluate_2021 --stage score      # GPU only, reads the CQT cache
    python -m src.evaluate_2021 --limit 2000       # smoke test
    python -m src.evaluate_2021 --verify-only      # byte-check the cache
    python -m src.evaluate_2021 --merge-only       # rebuild merged index/scores

THIS IS THE ONE PASS OVER THE HELD-OUT SET. The systems it scores were fixed in
config.PHASE7_LCNN_SYSTEMS before any 2021 number existed (PROJECT_PLAN.md phase 7);
nothing here selects, tunes or thresholds against 2021.

TWO STAGES, AND WHY -- this is the memory architecture, not a style choice.

The obvious design interleaves extraction and scoring so the corpus is read once.
It was built that way first and it FAILED: 89/8,000 files died with Windows error
1455 (ERROR_COMMITMENT_LIMIT) plus ffmpeg STATUS_COMMITMENT_LIMIT, and throughput
fell to 28 files/s as the machine paged. On 5.9GB of RAM the interleaved design asks
for a parent holding torch + a CUDA context + 7 models (~2GB) at the same time as 8
worker interpreters holding librosa/numba (~250MB each). Phase 4 ran 8 workers
safely precisely because its parent had no torch at all.

Starving the workers instead (6 jobs, no prefetch) did fix the failures, but at
18 files/s -- a 14.6h run.

Splitting the stages removes the contention rather than rationing it:

*   `extract` imports NO torch (the imports are function-local, and datasets.py
    pulls torch in too, so it is imported lazily as well). The parent is light, 8
    workers fit, and this is exactly Phase 4's proven configuration.
*   `score` runs no extraction workers. It reads CQT back from the cache at
    ~0.27 ms/sample (Phase 6's measurement for reads inside one open blob) and is
    pure GPU work.

The corpus is still decoded exactly once. This is only possible because the CQT is
cached -- which was justified independently as insurance (config.PHASE7_CACHE_CQT),
and now pays for itself a second time.

Other structure worth knowing:

*   The 7 LCNNs need only FOUR distinct inputs -- (T400, unit), (T400, cmvn),
    (T250, unit), (T150, unit) -- so each chunk's CQT is windowed four times, not
    seven. Four of the seven share the (T400, unit) view.

*   Windowing and normalisation come from datasets.py, the exact functions the
    Phase 6 dev evaluation used. A divergence there would silently invalidate the
    dev-vs-eval comparison this whole phase exists to make.

*   Resumable at chunk granularity, per stage. A chunk counts as done only when its
    shard holds a row for EVERY file in it -- not merely when the shard exists. An
    earlier version marked short shards complete and silently dropped 11% of the
    set while reporting success. Every write is a temp-file rename, so a crash
    mid-write cannot leave a truncated shard that resume would trust.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import BrokenProcessPool, TerminatedWorkerError
from tqdm import tqdm

from . import config
from .features import extract_cqt_uint8, extract_for_eval, load_audio

MFCC_COLS = [f"mfcc_{i}" for i in range(120)]
FAILURES_DIR = config.PA2021_WORK_DIR / "failures"
FAILURES_DIR.mkdir(parents=True, exist_ok=True)


# --- shard layout --------------------------------------------------------------

def shard_paths(chunk: int) -> tuple[Path, Path, Path, Path]:
    return (config.PA2021_CQT_SHARD_DIR / f"cqt_{chunk:05d}.dat",
            config.PA2021_CQT_SHARD_DIR / f"index_{chunk:05d}.parquet",
            config.PA2021_MFCC_SHARD_DIR / f"mfcc_{chunk:05d}.parquet",
            config.PA2021_SCORE_SHARD_DIR / f"scores_{chunk:05d}.parquet")


def _rows(path: Path) -> int:
    return len(pd.read_parquet(path, columns=["filename"]))


def _accepted(chunk: int) -> bool:
    """Set by --accept-failures once a chunk's failures are understood to be genuine
    undecodable files, so it is not retried forever."""
    return (config.PA2021_SCORE_SHARD_DIR / f"accepted_{chunk:05d}.flag").exists()


def extract_complete(chunk: int, expected: int) -> bool:
    blob, index, mfcc, _ = shard_paths(chunk)
    if not (blob.exists() and index.exists() and mfcc.exists()):
        return False
    return _accepted(chunk) or _rows(index) == expected


def score_complete(chunk: int) -> bool:
    """Scored rows must match EXTRACTED rows, not the original chunk size -- a file
    that failed extraction has nothing to score and must not block the chunk."""
    _, index, _, score = shard_paths(chunk)
    if not (index.exists() and score.exists()):
        return False
    return _rows(score) == _rows(index)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --- stage 1: extract ----------------------------------------------------------

def recycle_workers() -> None:
    """Tear the loky pool down so the next Parallel() spawns fresh worker processes.

    loky REUSES one pool across every Parallel() call, which is normally what you
    want -- but over hundreds of chunks those 8 long-lived interpreters accumulate
    heap fragmentation and OS handles until the system cannot service a large pipe
    write. That is not hypothetical: a run died after 117 chunks (468,000 files) with
    WinError 1450 (ERROR_NO_SYSTEM_RESOURCES) raised from loky's _sendback_result
    while returning a ~54MB chunk. Recycling costs a few seconds of respawn and
    resets that accumulation to zero.
    """
    try:
        get_reusable_executor().shutdown(wait=True)
    except Exception:                                     # noqa: BLE001
        pass


def extract_chunk(records: list[dict], n_jobs: int, attempts: int = 3) -> list[dict]:
    """Extract one chunk, retrying on worker-pool failures.

    The worker is features.extract_for_eval, deliberately defined in a torch-free
    module; loky pickles it by reference, so workers import src.features only.

    Per-FILE errors are caught inside the worker and returned as rows, so anything
    raised here is a failure of the pool itself -- a broken/terminated worker or an
    OS resource error in the result transport. Those are transient and recoverable
    by rebuilding the pool, and must not be allowed to kill a multi-hour run that is
    otherwise succeeding. A chunk is only ever written whole, so a retry cannot
    produce partial output.
    """
    for attempt in range(1, attempts + 1):
        try:
            return Parallel(n_jobs=n_jobs)(delayed(extract_for_eval)(r) for r in records)
        except (OSError, TerminatedWorkerError, BrokenProcessPool) as e:
            if attempt == attempts:
                raise
            tqdm.write(f"  worker pool failed ({type(e).__name__}: {str(e)[:90]}); "
                       f"recycling and retrying chunk (attempt {attempt + 1}/{attempts})")
            recycle_workers()
            time.sleep(10)   # let the OS reclaim handles/pool before respawning
    raise AssertionError("unreachable")


def write_extract_chunk(chunk: int, ok: list[dict], meta: pd.DataFrame) -> None:
    """Blob first, then index, then MFCC. Variable-length records, so each offset is
    data and must be stored -- position cannot be recomputed arithmetically."""
    blob_p, index_p, mfcc_p, _ = shard_paths(chunk)
    names = [r["filename"] for r in ok]
    m = meta.set_index("filename").loc[names]

    tmp = blob_p.with_name(blob_p.name + ".tmp")
    offsets, off = [], 0
    with open(tmp, "wb") as fh:
        for r in ok:
            fh.write(np.ascontiguousarray(r["cqt"]).tobytes())
            offsets.append(off)
            off += r["cqt"].nbytes
    os.replace(tmp, blob_p)

    _atomic_parquet(pd.DataFrame({
        "filename": names, "shard": chunk, "offset": offsets,
        "n_frames": [r["cqt"].shape[1] for r in ok],
        "duration_s": [r["n_samples"] / config.SAMPLE_RATE for r in ok],
        "label": m["label"].to_numpy(), "partition": m["partition"].to_numpy(),
    }), index_p)

    mfcc_df = pd.DataFrame(np.stack([r["mfcc"] for r in ok]), columns=MFCC_COLS)
    mfcc_df.insert(0, "filename", names)
    _atomic_parquet(mfcc_df, mfcc_p)


def run_extract(files: pd.DataFrame, starts: list[int], sizes: list[int],
                args) -> None:
    todo = [c for c in range(len(starts)) if not extract_complete(c, sizes[c])]
    print(f"\n=== stage 1/2: extract ===  {len(starts) - len(todo)}/{len(starts)} "
          f"chunks already done, {sum(sizes[c] for c in todo):,} files to go")
    if not todo:
        return

    t0, n_done, n_failed = time.time(), 0, 0
    bar = tqdm(todo, desc="extract", unit="chunk")
    for i, c in enumerate(bar):
        # Proactively rebuild the pool well before the resource exhaustion that
        # killed a run at chunk 117 (see recycle_workers). A few seconds of respawn
        # every N chunks is cheap insurance against losing hours of progress.
        if i and args.recycle_every and i % args.recycle_every == 0:
            recycle_workers()

        records = files.iloc[starts[c]:starts[c] + sizes[c]].to_dict("records")
        results = extract_chunk(records, args.n_jobs)

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
            write_extract_chunk(c, ok, files)
        if bad and args.accept_failures:
            (config.PA2021_SCORE_SHARD_DIR / f"accepted_{c:05d}.flag").write_text(
                f"{len(bad)} permanently-failed files accepted\n", encoding="utf-8")

        n_done += len(results)
        del results, ok, bad

        # Abort loudly on a systemic failure rather than grinding through the corpus
        # dropping files. Phase 4 decoded 241,056 files through this same ffmpeg path
        # with ZERO failures, so a sustained rate here means something environmental
        # that re-running will not fix and that must not reach a headline number.
        if n_failed > max(5, args.max_failure_rate * n_done):
            raise RuntimeError(
                f"{n_failed}/{n_done} files failed to extract "
                f"({n_failed / n_done:.1%}), above --max-failure-rate "
                f"{args.max_failure_rate:.1%}. Inspect {FAILURES_DIR}; completed "
                f"chunks are kept, so fixing the cause and re-running resumes here.")

        rate = n_done / max(time.time() - t0, 1e-9)
        bar.set_postfix(files=f"{n_done:,}", rate=f"{rate:.0f}/s",
                        eta=f"{sum(sizes[x] for x in todo[i+1:]) / max(rate, 1e-9) / 3600:.1f}h",
                        fail=n_failed)
    print(f"  extracted {n_done:,} files in {(time.time()-t0)/3600:.2f}h, "
          f"{n_failed} failures")


# --- stage 2: score ------------------------------------------------------------

def load_systems(tags: list[str], device):
    """Rebuild each pre-registered LCNN from its own checkpoint.

    Architecture comes from the `args` dict saved beside the weights, never from
    config or the tag string: a model silently rebuilt at the wrong T or head would
    still load and still produce plausible scores.
    """
    import torch
    from .models_lcnn import LCNN

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
                        "head": a["head"], "dev_eer": float(ck["best_eer"])}
    return systems


def group_views(systems: dict) -> dict:
    """Map (T, norm) -> [tags], so each distinct input is built once per chunk."""
    views: dict[tuple[int, str], list[str]] = {}
    for tag, s in systems.items():
        views.setdefault((s["T"], s["norm"]), []).append(tag)
    return views


def score_chunk(systems: dict, views: dict, cqts: list[np.ndarray],
                device, batch: int) -> dict[str, np.ndarray]:
    """Window each chunk once per view and run every model sharing that view.

    Batched rather than whole-chunk because 4,000 files at T=400 in float32 would be
    576MB per view on a 5.9GB machine; at batch 64 it is 9.2MB.
    """
    import torch
    from .datasets import fit_length, normalise

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for (T, norm), tags in views.items():
            acc = {t: [] for t in tags}
            for i in range(0, len(cqts), batch):
                arr = np.stack([normalise(fit_length(c, T), norm)
                                for c in cqts[i:i + batch]])
                x = torch.from_numpy(arr).unsqueeze(1).to(device, non_blocking=True)
                for t in tags:
                    acc[t].append(systems[t]["model"](x).float().cpu().numpy())
            for t in tags:
                out[t] = np.concatenate(acc[t])
    return out


def read_chunk_cqt(chunk: int) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Sequential reads inside ONE open blob -- Phase 6 measured this at 0.27ms/
    sample versus 10.3ms for a per-file open, which is why the cache is a blob."""
    blob_p, index_p, _, _ = shard_paths(chunk)
    idx = pd.read_parquet(index_p)
    arrs = []
    with open(blob_p, "rb", buffering=0) as fh:
        for off, nf in zip(idx["offset"].to_numpy(), idx["n_frames"].to_numpy()):
            fh.seek(int(off))
            raw = fh.read(config.CQT_N_BINS * int(nf))
            arrs.append(np.frombuffer(raw, dtype=np.uint8).reshape(
                config.CQT_N_BINS, int(nf)))
    return idx, arrs


def run_score(starts: list[int], args) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    systems = load_systems(list(config.PHASE7_LCNN_SYSTEMS), device)
    views = group_views(systems)

    todo = [c for c in range(len(starts))
            if shard_paths(c)[1].exists() and not score_complete(c)]
    print(f"\n=== stage 2/2: score ===  device {device}, "
          f"{len(systems)} LCNNs in {len(views)} views, {len(todo)} chunks to score")
    for (T, norm), tags in sorted(views.items()):
        print(f"    T={T:<4} norm={norm:<5} -> {', '.join(tags)}")
    if not todo:
        return

    t0, n_done = time.time(), 0
    bar = tqdm(todo, desc="score", unit="chunk")
    for c in bar:
        idx, cqts = read_chunk_cqt(c)
        scores = score_chunk(systems, views, cqts, device, args.batch_size)
        out = pd.DataFrame({"filename": idx["filename"].to_numpy(),
                            "n_frames": idx["n_frames"].to_numpy(),
                            "duration_s": idx["duration_s"].to_numpy()})
        for tag, v in scores.items():
            out[tag] = v.astype(np.float32)
        _atomic_parquet(out, shard_paths(c)[3])
        n_done += len(idx)
        del idx, cqts, scores, out
        rate = n_done / max(time.time() - t0, 1e-9)
        bar.set_postfix(files=f"{n_done:,}", rate=f"{rate:.0f}/s")
    print(f"  scored {n_done:,} files in {(time.time()-t0)/3600:.2f}h")


# --- merge / verify ------------------------------------------------------------

def merge_shards(quiet: bool = False) -> pd.DataFrame:
    score_files = sorted(config.PA2021_SCORE_SHARD_DIR.glob("scores_*.parquet"))
    if not score_files:
        raise RuntimeError("no score shards to merge -- run the scoring stage first")
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
        print(f"\nmerged {len(score_files)} score shards -> {len(scores):,} rows "
              f"at {config.PA2021_LCNN_SCORES}")
        if fail_files:
            print(f"{len(fails):,} extraction failures -> {config.PA2021_FAILURES_CSV}")
    return scores


def verify_cache(n: int, manifest: pd.DataFrame) -> bool:
    """Re-extract random files from source and compare BYTE-FOR-BYTE to the cache.

    Same check pack_features.py runs, for the same reason: a wrong offset would
    corrupt every score downstream while surfacing no error anywhere.
    """
    if not config.PA2021_CQT_INDEX.exists():
        print("no merged CQT index -- nothing to verify")
        return False
    idx = pd.read_parquet(config.PA2021_CQT_INDEX)
    paths = manifest.set_index("filename")["filepath"]
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = idx.iloc[rng.choice(len(idx), min(n, len(idx)), replace=False)]

    bad = 0
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="verify", unit="file"):
        blob = config.PA2021_CQT_SHARD_DIR / f"cqt_{int(row['shard']):05d}.dat"
        with open(blob, "rb", buffering=0) as fh:
            fh.seek(int(row["offset"]))
            raw = fh.read(config.CQT_N_BINS * int(row["n_frames"]))
        cached = np.frombuffer(raw, dtype=np.uint8).reshape(
            config.CQT_N_BINS, int(row["n_frames"]))
        if not np.array_equal(extract_cqt_uint8(load_audio(paths[row["filename"]])), cached):
            bad += 1
            print(f"  MISMATCH: {row['filename']}")
    print(f"verify: {len(sample) - bad}/{len(sample)} byte-identical to a fresh extraction")
    return bad == 0


# --- main ----------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["both", "extract", "score"], default="both")
    p.add_argument("--chunk-size", type=int, default=config.PHASE7_CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=config.PHASE7_EVAL_BATCH)
    p.add_argument("--n-jobs", type=int, default=config.FEATURE_EXTRACTION_N_JOBS)
    p.add_argument("--recycle-every", type=int, default=config.PHASE7_RECYCLE_EVERY,
                   help="rebuild the worker pool every N chunks (0 disables)")
    p.add_argument("--limit", type=int, default=0, help="only the first N files (smoke test)")
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
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    manifest = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet")
    files = manifest[manifest["partition"].isin(config.PHASE7_PARTITIONS)]
    # Sort by filename so chunk boundaries -- and therefore shard contents -- are
    # identical on every run, which is what makes resume-by-shard-contents sound.
    files = files.sort_values("filename").reset_index(drop=True)
    if args.limit:
        files = files.iloc[:args.limit].reset_index(drop=True)

    if args.merge_only:
        merge_shards()
        return
    if args.verify_only:
        sys.exit(0 if verify_cache(args.verify_n, files) else 1)

    starts = list(range(0, len(files), args.chunk_size))
    sizes = [min(args.chunk_size, len(files) - s) for s in starts]

    print(f"Phase 7 -- the HELD-OUT 2021 PA set, stage={args.stage}")
    print(f"  partitions {list(config.PHASE7_PARTITIONS)}: {len(files):,} files "
          f"in {len(starts)} chunks of {args.chunk_size}")

    if args.stage in ("both", "extract"):
        run_extract(files, starts, sizes, args)
    if args.stage in ("both", "score"):
        run_score(starts, args)

    if args.stage in ("both", "score"):
        scores = merge_shards()
        missing = set(files["filename"]) - set(scores["filename"])
        if missing:
            print(f"WARNING: {len(missing):,} files have no score (see "
                  f"{config.PA2021_FAILURES_CSV}); every reported metric covers the rest")
        if args.verify_n:
            verify_cache(args.verify_n, files)


if __name__ == "__main__":
    main()
