"""Phase 1: turn the scattered protocol .txt files into clean pandas manifests.

Produces:
    manifests/pa2019_cm.parquet     -- 2019 CM-labeled train/dev/eval (218,430 rows)
    manifests/pa2019_asv_enroll.parquet -- 2019 ASV-enrollment-only bonafide files (22,626 rows)
    manifests/pa2021_cm.parquet     -- 2021 CM eval, all partitions (943,110 rows)
"""
from __future__ import annotations

import pandas as pd

from . import config


def load_2019_cm_protocol(path, split: str, flac_dir) -> pd.DataFrame:
    cols = ["speaker_id", "filename", "env_id", "attack_id", "label"]
    df = pd.read_csv(path, sep=" ", names=cols)
    df["filepath"] = df["filename"].apply(lambda f: str(flac_dir / "flac" / f"{f}.flac"))
    df["split"] = split
    df["year"] = 2019
    return df


def build_pa2019_cm_manifest() -> pd.DataFrame:
    frames = [
        load_2019_cm_protocol(
            config.PA2019_CM_PROTOCOL_FILES[split],
            split,
            config.PA2019_SPLIT_DIRS[split],
        )
        for split in ("train", "dev", "eval")
    ]
    return pd.concat(frames, ignore_index=True)


def build_pa2019_asv_enroll_manifest() -> pd.DataFrame:
    rows = []
    for path in config.PA2019_ASV_ENROLL_FILES:
        split = config.PA2019_ASV_ENROLL_SPLIT[path.name]
        flac_dir = config.PA2019_SPLIT_DIRS[split] / "flac"
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                speaker_env, file_list = line.split(" ", 1)
                speaker_id = speaker_env.rsplit("_", 1)[0]
                for filename in file_list.split(","):
                    rows.append(
                        {
                            "speaker_id": speaker_id,
                            "filename": filename,
                            "env_id": speaker_env.rsplit("_", 1)[1],
                            "attack_id": "-",
                            "label": "bonafide",
                            "filepath": str(flac_dir / f"{filename}.flac"),
                            "split": split,
                            "year": 2019,
                            "source": "asv_enrollment",
                        }
                    )
    return pd.DataFrame(rows)


def build_pa2021_filename_to_path() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for part_dir in config.PA2021_PART_DIRS:
        for flac_path in part_dir.glob("*.flac"):
            lookup[flac_path.stem] = str(flac_path)
    return lookup


def build_pa2021_cm_manifest() -> pd.DataFrame:
    cols = [
        "speaker_id",
        "filename",
        "room",
        "mic",
        "dist",
        "r",
        "m",
        "s",
        "c",
        "label",
        "trim_flag",
        "partition",
    ]
    df = pd.read_csv(config.PA2021_CM_KEYS_FILE, sep=" ", names=cols)

    lookup = build_pa2021_filename_to_path()
    df["filepath"] = df["filename"].map(lookup)
    df["year"] = 2021

    missing = df["filepath"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} filenames in trial_metadata.txt have no matching flac file")

    return df


def main() -> None:
    print("Building 2019 CM manifest...")
    pa2019_cm = build_pa2019_cm_manifest()
    pa2019_cm.to_parquet(config.MANIFESTS_DIR / "pa2019_cm.parquet", index=False)
    print(f"  {len(pa2019_cm)} rows -> manifests/pa2019_cm.parquet")
    print(pa2019_cm["label"].value_counts().to_dict())

    print("Building 2019 ASV-enrollment manifest...")
    pa2019_asv_enroll = build_pa2019_asv_enroll_manifest()
    pa2019_asv_enroll.to_parquet(config.MANIFESTS_DIR / "pa2019_asv_enroll.parquet", index=False)
    print(f"  {len(pa2019_asv_enroll)} rows -> manifests/pa2019_asv_enroll.parquet")

    print("Building 2021 CM manifest (this scans 943,110 flac files across 7 parts, may take a while)...")
    pa2021_cm = build_pa2021_cm_manifest()
    pa2021_cm.to_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet", index=False)
    print(f"  {len(pa2021_cm)} rows -> manifests/pa2021_cm.parquet")
    print(pa2021_cm["partition"].value_counts().to_dict())


if __name__ == "__main__":
    main()
