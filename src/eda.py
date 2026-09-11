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

import argparse
import random
import shutil
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
SLIKE_SR_DIR = config.PROJECT_ROOT / "Diplomski rad" / "slike"

sns.set_theme(style="whitegrid")
RNG_SEED = config.RANDOM_SEED

# --- language -------------------------------------------------------------------
# The submitted thesis is Serbian Cyrillic; --lang sr redraws the figures it uses
# with Serbian labels and writes them as <name>_srp.png, leaving the English
# originals in place. Terms that stay English (MFCC, CQT, STFT, PA, the partition
# names, speaker and file ids) simply have no entry here and fall through.
LANG = "en"

SR = {
    "Duration (s)": "трајање (s)",
    "Time (s)": "време (s)",
    "Waveform": "сигнал у временском домену",
    "STFT spectrogram (Hz)": "STFT спектрограм (Hz)",
    "CQTgram (Hz)": "CQT спектрограм (Hz)",
    "MFCC coefficient": "MFCC коефицијент",
    "Bonafide": "истински",
    "Spoof": "лажиран",
    "bonafide": "истински",
    "spoof": "лажиран",
    "label": "класа",
    "dataset": "корпус",
    "2019 PA (train pool)": "2019 PA (скуп за обуку)",
    "2021 PA eval": "2021 PA eval",
    "2019 PA duration by label": "2019 PA трајање по класи",
    "2019 vs. 2021 duration distribution (sampled)":
        "расподела трајања, 2019 наспрам 2021",
    "Count": "број снимака",
    "Density": "густина",
}


def _t(s: str) -> str:
    return SR.get(s, s) if LANG == "sr" else s


def _fname(name: str) -> str:
    """Output filename for the language being drawn."""
    return name.replace(".png", "_srp.png") if LANG == "sr" else name


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

def sample_durations(df: pd.DataFrame, n_per_label: int, seed: int = RNG_SEED,
                     cache: str | None = None) -> pd.DataFrame:
    """Durations of a fixed random sample, from the file headers.

    Cached to EDA/ because it is the one expensive step in this module -- a few
    thousand header reads off the corpus drive -- and redrawing the same figure in
    another language should not pay for it twice. The sample is seeded, so the
    cache holds exactly what a re-read would produce.
    """
    if cache is not None:
        path = EDA_DIR / cache
        if path.exists():
            return pd.read_csv(path)
    rows = []
    for label, grp in df.groupby("label"):
        sample = grp.sample(n=min(n_per_label, len(grp)), random_state=seed)
        for p in tqdm(sample["filepath"], desc=f"duration:{label}", leave=False):
            info = sf.info(p)
            rows.append({"label": label, "duration_sec": info.frames / info.samplerate})
    out = pd.DataFrame(rows)
    if cache is not None:
        out.to_csv(EDA_DIR / cache, index=False)
    return out


def plot_duration_histograms(pool_after: pd.DataFrame, pa2021: pd.DataFrame):
    dur_2019 = sample_durations(pool_after, n_per_label=1500,
                                cache="03_durations_2019.csv")
    dur_2019["dataset"] = _t("2019 PA (train pool)")

    pa2021_eval = pa2021[pa2021["partition"] == config.PA2021_REPORTED_PARTITION]
    dur_2021 = sample_durations(pa2021_eval, n_per_label=1500,
                                cache="03_durations_2021.csv")
    dur_2021["dataset"] = _t("2021 PA eval")

    for d in (dur_2019, dur_2021):
        d["label"] = d["label"].map(_t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(data=dur_2019, x="duration_sec", hue="label", bins=40, ax=axes[0], element="step")
    axes[0].set_title(_t("2019 PA duration by label")
                      if LANG == "sr" else
                      f"2019 PA duration by label (n={len(dur_2019)} sampled)")
    axes[0].set_xlabel(_t("Duration (s)"))
    if axes[0].get_legend():
        axes[0].get_legend().set_title(_t("label"))

    combined = pd.concat([dur_2019, dur_2021], ignore_index=True)
    sns.histplot(data=combined, x="duration_sec", hue="dataset", bins=40, ax=axes[1], element="step", stat="density", common_norm=False)
    axes[1].set_title(_t("2019 vs. 2021 duration distribution (sampled)"))
    axes[1].set_xlabel(_t("Duration (s)"))
    if axes[1].get_legend():
        axes[1].get_legend().set_title(_t("dataset"))
    axes[0].set_ylabel(_t("Count"))
    axes[1].set_ylabel(_t("Density"))

    fig.tight_layout()
    fig.savefig(EDA_DIR / _fname("03_duration_histograms.png"), dpi=150)
    plt.close(fig)

    # Canonical result, written only by the English run: a --lang sr pass would
    # otherwise translate the dataset names inside a persisted results file, which
    # is a figure concern leaking somewhere it does not belong.
    if LANG != "sr":
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
        f"{_t('Bonafide')}  ({bonafide_row['speaker_id']}, {bonafide_row['filename']})",
        f"{_t('Spoof')}  ({spoof_row['speaker_id']}, {spoof_row['attack_id']}, {spoof_row['filename']})",
    ]

    for col, (y, title) in enumerate(zip([y_bona, y_spoof], titles)):
        t = np.arange(len(y)) / config.SAMPLE_RATE
        axes[0, col].plot(t, y, linewidth=0.5)
        axes[0, col].set_title(title)
        axes[0, col].set_xlabel(_t("Time (s)"))
        if col == 0:
            axes[0, col].set_ylabel(_t("Waveform"))

        stft = librosa.stft(y, n_fft=config.MFCC_N_FFT, hop_length=config.MFCC_HOP_LENGTH)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        img = librosa.display.specshow(
            stft_db, sr=config.SAMPLE_RATE, hop_length=config.MFCC_HOP_LENGTH,
            x_axis="time", y_axis="hz", ax=axes[1, col],
        )

        cqt = librosa.cqt(
            y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        librosa.display.specshow(
            cqt_db, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            x_axis="time", y_axis="cqt_hz", bins_per_octave=config.CQT_BINS_PER_OCTAVE, ax=axes[2, col],
        )
        # specshow writes its own axis labels, in English ("Time", "Hz"), so both are
        # set again here. Each unit is named once, on the left: a second "Hz" on the
        # right column only sits between the two panels repeating it.
        for row in (1, 2):
            axes[row, col].set_xlabel(_t("Time (s)"))
        axes[1, col].set_ylabel(_t("STFT spectrogram (Hz)") if col == 0 else "")
        axes[2, col].set_ylabel(_t("CQTgram (Hz)") if col == 0 else "")

    # No suptitle in Serbian: the Typst caption already names the figure, and a
    # sentence across the top only repeats it.
    if LANG != "sr":
        fig.suptitle("Waveform / STFT spectrogram / CQTgram: bonafide vs. replayed speech (same speaker)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / _fname("04_waveform_spectrogram_cqt.png"), dpi=150)
    plt.close(fig)


def plot_mfcc_vs_cqt(bonafide_row, spoof_row):
    """Ties directly to the thesis's central claim (PROJECT_PLAN.md section 6):
    the mel filterbank in MFCC compresses exactly the high-frequency region where
    loudspeaker/microphone artifacts live, while CQT preserves it.
    """
    y_bona = _load_audio(bonafide_row["filepath"])
    y_spoof = _load_audio(spoof_row["filepath"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    cols = [(_t("Bonafide"), y_bona), (_t("Spoof"), y_spoof)]

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
        axes[0, col].set_title(
            f"{label}: MFCC ({config.N_MFCC} коефицијената)" if LANG == "sr"
            else f"{label}: MFCC ({config.N_MFCC} coeffs, z-scored per coeff for display)")
        axes[0, col].set_yticks(range(0, config.N_MFCC, 4))

        cqt = librosa.cqt(
            y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        librosa.display.specshow(
            cqt_db, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
            x_axis="time", y_axis="cqt_hz", bins_per_octave=config.CQT_BINS_PER_OCTAVE, ax=axes[1, col],
        )
        axes[1, col].set_title(
            f"{label}: CQT ({config.CQT_N_BINS} опсега)" if LANG == "sr"
            else f"{label}: CQTgram ({config.CQT_N_BINS} bins)")
        # As in the figure above: specshow's English labels are replaced, and each
        # unit is named once, on the left column. The CQT row names the transform
        # too, since "Hz" alone does not say what is being shown.
        for row in (0, 1):
            axes[row, col].set_xlabel(_t("Time (s)"))
        axes[0, col].set_ylabel(_t("MFCC coefficient") if col == 0 else "")
        axes[1, col].set_ylabel(_t("CQTgram (Hz)") if col == 0 else "")

    if LANG != "sr":
        fig.suptitle("MFCC vs. CQT: does the replay fingerprint survive the front-end?")
    fig.tight_layout()
    fig.savefig(EDA_DIR / _fname("05_mfcc_vs_cqt.png"), dpi=150)
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", choices=("en", "sr"), default="en",
                    help="label language; 'sr' writes <name>_srp.png alongside the "
                         "English originals")
    ap.add_argument("--only", nargs="+", type=int, choices=range(1, 8), metavar="N",
                    help="generate only these plot numbers (1-7); the Serbian thesis "
                         "uses 3, 4 and 5")
    args = ap.parse_args()

    global LANG
    LANG = args.lang
    want = set(args.only) if args.only else set(range(1, 8))

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    print(f"Loading manifests/splits... (lang={LANG})")
    cm_before, pool_after, train_df, dev_df, pa2021 = load_data()

    if 1 in want:
        print("Plot 1/7: class balance before/after enrichment")
        plot_class_balance_before_after(cm_before, pool_after)

    if 2 in want:
        print("Plot 2/7: class balance train_2019 vs dev_2019")
        plot_class_balance_train_dev(train_df, dev_df)

    if 3 in want:
        print("Plot 3/7: duration histograms (2019 by label, 2019 vs 2021)")
        plot_duration_histograms(pool_after, pa2021)

    if want & {4, 5}:
        print("Picking a same-speaker bonafide/spoof example pair...")
        bonafide_row, spoof_row = pick_example_pair(train_df)
        print(f"  bonafide: {bonafide_row['filepath']}")
        print(f"  spoof:    {spoof_row['filepath']} (attack {spoof_row['attack_id']})")

        if 4 in want:
            print("Plot 4/7: waveform / STFT spectrogram / CQTgram comparison")
            plot_waveform_spectrogram_cqt(bonafide_row, spoof_row)

        if 5 in want:
            print("Plot 5/7: MFCC vs CQT comparison (thesis's central empirical argument)")
            plot_mfcc_vs_cqt(bonafide_row, spoof_row)

    if 6 in want:
        print("Plot 6/7: speaker count and gender balance per split")
        plot_speaker_and_gender_balance(train_df, dev_df, build_gender_map())

    if 7 in want:
        print("Plot 7/7: attack-condition distribution sanity check")
        plot_attack_condition_distribution(train_df, dev_df)

    # The Serbian thesis builds from its own slike/ folder, so a --lang sr run puts
    # its figures there too, the way plot_thesis does for the results figures.
    if LANG == "sr" and SLIKE_SR_DIR.exists():
        for n in sorted(want):
            for p in sorted(EDA_DIR.glob(f"{n:02d}_*_srp.png")):
                shutil.copyfile(p, SLIKE_SR_DIR / p.name)
                print(f"  mirrored {p.name} -> {SLIKE_SR_DIR.parent.name}/slike/")

    print(f"\nAll EDA outputs written to {EDA_DIR}")


if __name__ == "__main__":
    main()
