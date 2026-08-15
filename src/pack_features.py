"""Phase 6 prerequisite: pack the per-file CQT cache into one contiguous blob per split.

Why this exists (measured, not assumed):

    reading 175,959 individual .npy files, COLD
        np.load per file          10.32 ms  ->  30.3 min per epoch, 1 worker
        np.load(mmap_mode='r')    10.41 ms  ->  30.5 min   (NO benefit)
        random reads inside ONE already-open file
                                   0.27 ms  ->   0.8 min

The cost is per-file-open overhead (syscall + AV scan + .npy header parse), not data
transfer -- which is why mmap does not help but packing does. Left unpacked, data
loading would dominate training by an order of magnitude over the GPU work.

Layout: one flat uint8 blob holding every file's (90, n_frames) CQT written
C-contiguous back to back, plus a parquet index carrying byte offset, frame count,
label and speaker for each file. Reading sample i is then a seek + read of
90 * n_frames[i] bytes from an already-open handle.

The original per-file cache is left in place (E: has ample room) so the pack can be
rebuilt without re-running Phase 4.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config


def source_rows() -> dict[str, pd.DataFrame]:
    """train/dev split membership comes from Phase 2's csvs -- the same speaker-disjoint
    split every later phase uses. Never re-derived here."""
    out = {}
    for split, path in (("train", "train_2019.csv"), ("dev", "dev_2019.csv")):
        df = pd.read_csv(config.SPLITS_DIR / path)
        out[split] = df[["filename", "speaker_id", "label", "attack_id", "env_id"]].copy()
    return out


def pack_split(split: str, rows: pd.DataFrame, force: bool) -> pd.DataFrame:
    blob_path = config.PACKED_BLOB[split]
    index_path = config.PACKED_INDEX[split]

    if index_path.exists() and blob_path.exists() and not force:
        idx = pd.read_parquet(index_path)
        if len(idx) == len(rows):
            print(f"  {split}: already packed ({len(idx):,} files, "
                  f"{blob_path.stat().st_size/1e9:.2f} GB) -- skipping")
            return idx
        print(f"  {split}: index has {len(idx):,} rows but split has {len(rows):,} -- repacking")

    cqt_dir = config.FEATURES_DIR / "cqt"
    offsets, n_frames_list, kept = [], [], []
    offset = 0

    config.PACKED_DIR.mkdir(parents=True, exist_ok=True)
    with open(blob_path, "wb") as blob:
        for row in tqdm(rows.itertuples(index=False), total=len(rows),
                        desc=f"packing {split}", unit="file"):
            arr = np.load(cqt_dir / f"{row.filename}.npy")
            if arr.dtype != np.uint8 or arr.shape[0] != config.CQT_N_BINS:
                raise RuntimeError(
                    f"{row.filename}: expected uint8 ({config.CQT_N_BINS}, T), "
                    f"got {arr.dtype} {arr.shape}")
            arr = np.ascontiguousarray(arr)
            blob.write(arr.tobytes())
            offsets.append(offset)
            n_frames_list.append(arr.shape[1])
            kept.append(row)
            offset += arr.nbytes

    idx = pd.DataFrame(kept)
    idx["offset"] = offsets
    idx["n_frames"] = n_frames_list
    idx.to_parquet(index_path, index=False)

    print(f"  {split}: {len(idx):,} files -> {blob_path.name} "
          f"({offset/1e9:.2f} GB), index -> {index_path.name}")
    return idx


def verify(split: str, idx: pd.DataFrame, n_check: int = 200):
    """Read a random sample back out of the blob and compare against the original
    .npy byte for byte. A silent offset bug here would corrupt every batch of
    training with no visible error, so this is worth the few seconds."""
    cqt_dir = config.FEATURES_DIR / "cqt"
    rng = np.random.default_rng(config.RANDOM_SEED)
    picks = rng.choice(len(idx), min(n_check, len(idx)), replace=False)

    with open(config.PACKED_BLOB[split], "rb") as blob:
        for i in picks:
            r = idx.iloc[int(i)]
            blob.seek(int(r.offset))
            raw = blob.read(config.CQT_N_BINS * int(r.n_frames))
            got = np.frombuffer(raw, dtype=np.uint8).reshape(config.CQT_N_BINS, int(r.n_frames))
            want = np.load(cqt_dir / f"{r.filename}.npy")
            if not np.array_equal(got, want):
                raise RuntimeError(f"MISMATCH for {r.filename} at offset {r.offset}")
    print(f"  {split}: verified {len(picks)} random files byte-for-byte against the .npy cache")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    force = "--force" in sys.argv

    print("Packing CQT cache into contiguous blobs...")
    for split, rows in source_rows().items():
        idx = pack_split(split, rows, force)
        verify(split, idx)

    total = sum(config.PACKED_BLOB[s].stat().st_size for s in ("train", "dev"))
    print(f"\nDone. {total/1e9:.2f} GB total in {config.PACKED_DIR}")


if __name__ == "__main__":
    main()
