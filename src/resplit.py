"""Phase 2: reshuffle 2019 PA by speaker, enriched with ASV-enrollment bonafide files.

See PROJECT_PLAN.md section 4 for the full reasoning. 2019's own train/dev/eval CM
split is pooled and reshuffled -- 2019 no longer has (or needs) its own eval split,
since 2021 PA eval becomes the only true held-out test set, touched once at the end.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from . import config

TRAIN_FRACTION = 0.75  # fraction of *speakers*, not rows -- keeps train/dev speaker-disjoint


def build_enriched_pool() -> pd.DataFrame:
    cm = pd.read_parquet(config.MANIFESTS_DIR / "pa2019_cm.parquet")
    cm["source"] = "cm_protocol"

    enroll = pd.read_parquet(config.MANIFESTS_DIR / "pa2019_asv_enroll.parquet")

    return pd.concat([cm, enroll], ignore_index=True)


def speaker_disjoint_split(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, train_size=TRAIN_FRACTION, random_state=config.RANDOM_SEED
    )
    train_idx, dev_idx = next(splitter.split(pool, groups=pool["speaker_id"]))
    return pool.iloc[train_idx].reset_index(drop=True), pool.iloc[dev_idx].reset_index(drop=True)


def main() -> None:
    pool = build_enriched_pool()
    print(f"Enriched pool: {len(pool)} rows, {pool['speaker_id'].nunique()} speakers")
    print(pool["label"].value_counts().to_dict())

    train_df, dev_df = speaker_disjoint_split(pool)

    overlap = set(train_df["speaker_id"]) & set(dev_df["speaker_id"])
    if overlap:
        raise RuntimeError(f"Speaker leakage between train/dev: {overlap}")

    train_df.to_csv(config.SPLITS_DIR / "train_2019.csv", index=False)
    dev_df.to_csv(config.SPLITS_DIR / "dev_2019.csv", index=False)

    print(f"\ntrain_2019.csv: {len(train_df)} rows, {train_df['speaker_id'].nunique()} speakers")
    print(train_df["label"].value_counts().to_dict())
    print(f"\ndev_2019.csv: {len(dev_df)} rows, {dev_df['speaker_id'].nunique()} speakers")
    print(dev_df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
