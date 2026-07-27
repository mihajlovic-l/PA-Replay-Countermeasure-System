"""Phase 3: exploratory data analysis. Generates every plot into EDA/ (kept separate
from results/, which is reserved for actual model outputs -- EER tables, DET curves,
etc from Phase 7 onward).

Covers: class balance before/after the Phase 2 enrichment, class balance of the
resplit train/dev files, a duration histogram (2019 by label, and 2019 vs 2021 for
the domain-shift angle from PROJECT_PLAN.md section 3.3), a bonafide-vs-spoof
waveform/spectrogram comparison, an MFCC-vs-CQT high-frequency comparison (the
thesis's central empirical argument, see PROJECT_PLAN.md section 6), speaker count
and gender balance per split, and an attack-condition distribution sanity check.
"""
from __future__ import annotations

import random
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from tqdm import tqdm

from . import config, resplit

EDA_DIR = config.PROJECT_ROOT / "EDA"
EDA_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
RNG_SEED = config.RANDOM_SEED


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    cm_before = pd.read_parquet(config.MANIFESTS_DIR / "pa2019_cm.parquet")
    pool_after = resplit.build_enriched_pool()
    train_df = pd.read_csv(config.SPLITS_DIR / "train_2019.csv")
    dev_df = pd.read_csv(config.SPLITS_DIR / "dev_2019.csv")
    pa2021 = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet")
    return cm_before, pool_after, train_df, dev_df, pa2021


def build_gender_map() -> dict[str, str]:
    """Speaker -> gender, derived from the ASV enrollment .trn files (dev/eval only).

    Coverage is partial: only speakers enrolled for the ASV task have a known
    gender this way (58 of 107 total 2019 PA speakers). The rest -- including all
    20 original PA train speakers, who have no ASV enrollment file at all -- are
    reported as "unknown" rather than guessed.
    """
    gender_map: dict[str, str] = {}
    file_gender = {
        "ASVspoof2019.PA.asv.dev.female.trn.txt": "female",
        "ASVspoof2019.PA.asv.dev.male.trn.txt": "male",
        "ASVspoof2019.PA.asv.eval.female.trn.txt": "female",
        "ASVspoof2019.PA.asv.eval.male.trn.txt": "male",
    }
    for path in config.PA2019_ASV_ENROLL_FILES:
        gender = file_gender[path.name]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                speaker_env = line.split(" ", 1)[0]
                speaker_id = speaker_env.rsplit("_", 1)[0]
                gender_map[speaker_id] = gender
    return gender_map


# ---------------------------------------------------------------------------
# 1. Class balance before/after enrichment
# ---------------------------------------------------------------------------

def plot_class_balance_before_after(cm_before: pd.DataFrame, pool_after: pd.DataFrame):
    before_counts = cm_before["label"].value_counts()
    after_counts = pool_after["label"].value_counts()

    data = pd.DataFrame(
        {
            "Before enrichment\n(2019 CM only)": before_counts,
            "After enrichment\n(+ASV enrollment)": after_counts,
        }
    ).T.reset_index().melt(id_vars="index", var_name="label", value_name="count")
    data = data.rename(columns={"index": "stage"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=data, x="stage", y="count", hue="label", ax=ax)
    for stage, counts in (("Before enrichment\n(2019 CM only)", before_counts), ("After enrichment\n(+ASV enrollment)", after_counts)):
        ratio = counts["spoof"] / counts["bonafide"]
        ax.text(
            0 if "Before" in stage else 1,
            counts.max() * 1.03,
            f"spoof:bonafide = {ratio:.2f}:1",
            ha="center",
            fontsize=9,
        )
    ax.set_title("Class balance before vs. after Phase 2 enrichment")
    ax.set_xlabel("")
    ax.set_ylabel("File count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01_class_balance_before_after.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Class balance of the resplit train/dev files
# ---------------------------------------------------------------------------

def plot_class_balance_train_dev(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    data = pd.DataFrame(
        {
            "train_2019": train_df["label"].value_counts(),
            "dev_2019": dev_df["label"].value_counts(),
        }
    ).T.reset_index().melt(id_vars="index", var_name="label", value_name="count")
    data = data.rename(columns={"index": "split"})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=data, x="split", y="count", hue="label", ax=ax)
    ax.set_title("Class balance: resplit train_2019 vs. dev_2019")
    ax.set_xlabel("")
    ax.set_ylabel("File count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "02_class_balance_train_dev.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Duration histograms
# ---------------------------------------------------------------------------

def sample_durations(df: pd.DataFrame, n_per_label: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rows = []
    for label, grp in df.groupby("label"):
        sample = grp.sample(n=min(n_per_label, len(grp)), random_state=seed)
        for path in tqdm(sample["filepath"], desc=f"duration:{label}", leave=False):
            info = sf.info(path)
            rows.append({"label": label, "duration_sec": info.frames / info.samplerate})
    return pd.DataFrame(rows)


def plot_duration_histograms(pool_after: pd.DataFrame, pa2021: pd.DataFrame):
    dur_2019 = sample_durations(pool_after, n_per_label=1500)
    dur_2019["dataset"] = "2019 PA (train pool)"

    pa2021_eval = pa2021[pa2021["partition"] == config.PA2021_REPORTED_PARTITION]
    dur_2021 = sample_durations(pa2021_eval, n_per_label=1500)
    dur_2021["dataset"] = "2021 PA eval"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(data=dur_2019, x="duration_sec", hue="label", bins=40, ax=axes[0], element="step")
    axes[0].set_title(f"2019 PA duration by label (n={len(dur_2019)} sampled)")
    axes[0].set_xlabel("Duration (s)")

    combined = pd.concat([dur_2019, dur_2021], ignore_index=True)
    sns.histplot(data=combined, x="duration_sec", hue="dataset", bins=40, ax=axes[1], element="step", stat="density", common_norm=False)
    axes[1].set_title("2019 vs. 2021 duration distribution (sampled)")
    axes[1].set_xlabel("Duration (s)")

    fig.tight_layout()
    fig.savefig(EDA_DIR / "03_duration_histograms.png", dpi=150)
    plt.close(fig)

    summary = combined.groupby("dataset")["duration_sec"].describe()[["mean", "50%", "min", "max"]]
    summary.to_csv(EDA_DIR / "03_duration_summary.csv")


# ---------------------------------------------------------------------------
# 4. Waveform / spectrogram / CQTgram comparison, and 5. MFCC vs CQT comparison
# ---------------------------------------------------------------------------

def pick_example_pair(df: pd.DataFrame):
    bonafide_rows = df[df["label"] == "bonafide"]
    spoof_rows = df[df["label"] == "spoof"]
    speakers_with_both = set(bonafide_rows["speaker_id"]) & set(spoof_rows["speaker_id"])
    speaker = sorted(speakers_with_both)[0]

    bonafide_candidates = bonafide_rows[bonafide_rows["speaker_id"] == speaker]
    bonafide_row = bonafide_candidates.iloc[0]

    spoof_candidates = spoof_rows[spoof_rows["speaker_id"] == speaker]
    same_env = spoof_candidates[spoof_candidates["env_id"] == bonafide_row.get("env_id")]
    pool = same_env if len(same_env) else spoof_candidates
    mild_attack = pool[pool["attack_id"] == "AA"]
    spoof_row = mild_attack.iloc[0] if len(mild_attack) else pool.iloc[0]
    return bonafide_row, spoof_row


def _load_audio(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    assert sr == config.SAMPLE_RATE
    return y


def plot_waveform_spectrogram_cqt(bonafide_row, spoof_row):
    y_bona = _load_audio(bonafide_row["filepath"])
    y_spoof = _load_audio(spoof_row["filepath"])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    titles = [
        f"Bonafide  ({bonafide_row['speaker_id']}, {bonafide_row['filename']})",
        f"Spoof  ({spoof_row['speaker_id']}, attack {spoof_row['attack_id']}, {spoof_row['filename']})",
    ]

    for col, (y, title) in enumerate(zip([y_bona, y_spoof], titles)):
        t = np.arange(len(y)) / config.SAMPLE_RATE
        axes[0, col].plot(t, y, linewidth=0.5)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel("Time (s)")
        if col == 0:
            axes[0, col].set_ylabel("Waveform")

        stft = librosa.stft(y, n_fft=config.MFCC_N_FFT, hop_length=config.MFCC_HOP_LENGTH)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(
            stft_db, sr=config.SAMPLE_RATE, hop_length=config.MFCC_HOP_LENGTH,
            x_axis="time", y_axis="hz", ax=axes[1, col],
        )
        if col == 0:
            axes[1, col].set_ylabel("STFT spectrogram (Hz)")

        cqt = librosa.cqt(
            y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        librosa.display.specshow(
            cqt_db, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            x_axis="time", y_axis="cqt_hz", bins_per_octave=config.CQT_BINS_PER_OCTAVE, ax=axes[2, col],
        )
        if col == 0:
            axes[2, col].set_ylabel("CQTgram")

    fig.suptitle("Waveform / STFT spectrogram / CQTgram: bonafide vs. replayed speech (same speaker)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "04_waveform_spectrogram_cqt.png", dpi=150)
    plt.close(fig)


def plot_mfcc_vs_cqt(bonafide_row, spoof_row):
    """Ties directly to the thesis's central claim (PROJECT_PLAN.md section 6):
    the mel filterbank in MFCC compresses exactly the high-frequency region where
    loudspeaker/microphone artifacts live, while CQT preserves it.
    """
    y_bona = _load_audio(bonafide_row["filepath"])
    y_spoof = _load_audio(spoof_row["filepath"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cols = [("Bonafide", y_bona), ("Spoof", y_spoof)]

    for col, (label, y) in enumerate(cols):
        mfcc = librosa.feature.mfcc(
            y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
            n_fft=config.MFCC_N_FFT, hop_length=config.MFCC_HOP_LENGTH,
        )
        # Per-coefficient z-score for display only: the 0th coefficient (log-energy)
        # otherwise dominates the color scale and washes out the rest.
        mfcc_disp = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        img = librosa.display.specshow(
            mfcc_disp, sr=config.SAMPLE_RATE, hop_length=config.MFCC_HOP_LENGTH,
            x_axis="time", cmap="coolwarm", vmin=-3, vmax=3, ax=axes[0, col],
        )
        axes[0, col].set_title(f"{label}: MFCC ({config.N_MFCC} coeffs, z-scored per coeff for display)")
        axes[0, col].set_yticks(range(0, config.N_MFCC, 4))
        axes[0, col].set_ylabel("MFCC coefficient")

        cqt = librosa.cqt(
            y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        librosa.display.specshow(
            cqt_db, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            x_axis="time", y_axis="cqt_hz", bins_per_octave=config.CQT_BINS_PER_OCTAVE, ax=axes[1, col],
        )
        axes[1, col].set_title(f"{label}: CQTgram ({config.CQT_N_BINS} bins)")

    fig.suptitle("MFCC vs. CQT: does the replay fingerprint survive the front-end?")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "05_mfcc_vs_cqt.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Speaker count and 7. gender balance per split
# ---------------------------------------------------------------------------

def plot_speaker_and_gender_balance(train_df: pd.DataFrame, dev_df: pd.DataFrame, gender_map: dict[str, str]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    speaker_counts = pd.Series(
        {"train_2019": train_df["speaker_id"].nunique(), "dev_2019": dev_df["speaker_id"].nunique()}
    )
    sns.barplot(x=speaker_counts.index, y=speaker_counts.values, ax=axes[0])
    axes[0].set_title("Unique speakers per split")
    axes[0].set_ylabel("Speaker count")
    for i, v in enumerate(speaker_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha="center")

    rows = []
    for split_name, df in (("train_2019", train_df), ("dev_2019", dev_df)):
        speakers = df["speaker_id"].unique()
        for spk in speakers:
            rows.append({"split": split_name, "gender": gender_map.get(spk, "unknown")})
    gender_df = pd.DataFrame(rows)
    gender_pivot = gender_df.groupby(["split", "gender"]).size().unstack(fill_value=0)
    gender_pivot = gender_pivot.reindex(columns=["male", "female", "unknown"], fill_value=0)
    gender_pivot.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Speaker gender per split\n(unknown = no ASV-enrollment record, e.g. original PA train speakers)")
    axes[1].set_ylabel("Speaker count")
    axes[1].legend(title="gender")
    axes[1].tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(EDA_DIR / "06_speaker_gender_balance.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Attack-condition distribution sanity check
# ---------------------------------------------------------------------------

def plot_attack_condition_distribution(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    rows = []
    for split_name, df in (("train_2019", train_df), ("dev_2019", dev_df)):
        spoof = df[df["label"] == "spoof"]
        counts = spoof["attack_id"].value_counts(normalize=True).sort_index()
        for attack_id, frac in counts.items():
            rows.append({"split": split_name, "attack_id": attack_id, "fraction": frac})
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=data, x="attack_id", y="fraction", hue="split", ax=ax)
    ax.set_title("Spoof attack-condition distribution: train_2019 vs. dev_2019\n(sanity check -- all 9 combinations should appear in both, roughly balanced)")
    ax.set_xlabel("Attack ID (attacker-distance x replay-device-quality)")
    ax.set_ylabel("Fraction of spoof files")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "07_attack_condition_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console cp1252 can't print the Cyrillic project path
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    print("Loading manifests/splits...")
    cm_before, pool_after, train_df, dev_df, pa2021 = load_data()
    gender_map = build_gender_map()

    print("Plot 1/7: class balance before/after enrichment")
    plot_class_balance_before_after(cm_before, pool_after)

    print("Plot 2/7: class balance train_2019 vs dev_2019")
    plot_class_balance_train_dev(train_df, dev_df)

    print("Plot 3/7: duration histograms (2019 by label, 2019 vs 2021)")
    plot_duration_histograms(pool_after, pa2021)

    print("Picking a same-speaker bonafide/spoof example pair...")
    bonafide_row, spoof_row = pick_example_pair(train_df)
    print(f"  bonafide: {bonafide_row['filepath']}")
    print(f"  spoof:    {spoof_row['filepath']} (attack {spoof_row['attack_id']})")

    print("Plot 4/7: waveform / STFT spectrogram / CQTgram comparison")
    plot_waveform_spectrogram_cqt(bonafide_row, spoof_row)

    print("Plot 5/7: MFCC vs CQT comparison (thesis's central empirical argument)")
    plot_mfcc_vs_cqt(bonafide_row, spoof_row)

    print("Plot 6/7: speaker count and gender balance per split")
    plot_speaker_and_gender_balance(train_df, dev_df, gender_map)

    print("Plot 7/7: attack-condition distribution sanity check")
    plot_attack_condition_distribution(train_df, dev_df)

    print(f"\nAll EDA outputs written to {EDA_DIR}")


if __name__ == "__main__":
    main()
