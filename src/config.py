"""Single source of truth for paths and hyperparameters."""
from pathlib import Path

# --- Roots ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("E:/ASVspoof data")

MANIFESTS_DIR = PROJECT_ROOT / "manifests"
SPLITS_DIR = PROJECT_ROOT / "splits"
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for d in (MANIFESTS_DIR, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- ASVspoof2019 PA ---
PA2019_ROOT = DATA_ROOT / "ASVspoof2019_PA" / "PA"
PA2019_CM_PROTOCOLS = PA2019_ROOT / "ASVspoof2019_PA_cm_protocols"
PA2019_ASV_PROTOCOLS = PA2019_ROOT / "ASVspoof2019_PA_asv_protocols"

PA2019_SPLIT_DIRS = {
    "train": PA2019_ROOT / "ASVspoof2019_PA_train",
    "dev": PA2019_ROOT / "ASVspoof2019_PA_dev",
    "eval": PA2019_ROOT / "ASVspoof2019_PA_eval",
}

PA2019_CM_PROTOCOL_FILES = {
    "train": PA2019_CM_PROTOCOLS / "ASVspoof2019.PA.cm.train.trn.txt",
    "dev": PA2019_CM_PROTOCOLS / "ASVspoof2019.PA.cm.dev.trl.txt",
    "eval": PA2019_CM_PROTOCOLS / "ASVspoof2019.PA.cm.eval.trl.txt",
}

# ASV enrollment (.trn) files -- these are the "free bonus" bonafide-only
# recordings not used anywhere in the CM protocol (see PROJECT_PLAN.md section 3.1/4).
PA2019_ASV_ENROLL_FILES = [
    PA2019_ASV_PROTOCOLS / "ASVspoof2019.PA.asv.dev.female.trn.txt",
    PA2019_ASV_PROTOCOLS / "ASVspoof2019.PA.asv.dev.male.trn.txt",
    PA2019_ASV_PROTOCOLS / "ASVspoof2019.PA.asv.eval.female.trn.txt",
    PA2019_ASV_PROTOCOLS / "ASVspoof2019.PA.asv.eval.male.trn.txt",
]
# which flac dir each enrollment file's audio actually lives in (dev vs eval)
PA2019_ASV_ENROLL_SPLIT = {
    "ASVspoof2019.PA.asv.dev.female.trn.txt": "dev",
    "ASVspoof2019.PA.asv.dev.male.trn.txt": "dev",
    "ASVspoof2019.PA.asv.eval.female.trn.txt": "eval",
    "ASVspoof2019.PA.asv.eval.male.trn.txt": "eval",
}

# --- ASVspoof2021 PA eval ---
PA2021_ROOT = DATA_ROOT / "ASVspoof2021_PA_eval"
PA2021_PART_DIRS = [
    PA2021_ROOT / f"ASVspoof2021_PA_eval_part0{i}" / "ASVspoof2021_PA_eval" / "flac"
    for i in range(7)
]
PA2021_PROTOCOL_FILE = (
    PA2021_ROOT
    / "ASVspoof2021_PA_eval_part06"
    / "ASVspoof2021_PA_eval"
    / "ASVspoof2021.PA.cm.eval.trl.txt"
)
PA2021_KEYS_ROOT = PA2021_ROOT / "PA-keys-full" / "keys" / "PA"
PA2021_CM_KEYS_FILE = PA2021_KEYS_ROOT / "CM" / "trial_metadata.txt"
PA2021_ASV_KEYS_FILE = PA2021_KEYS_ROOT / "ASV" / "trial_metadata.txt"

PA2021_BASELINE_SCORE_FILES = {
    "CQCC-GMM": PA2021_KEYS_ROOT / "CM" / "CQCC-GMM" / "score.txt",
    "LFCC-GMM": PA2021_KEYS_ROOT / "CM" / "LFCC-GMM" / "score.txt",
    "LFCC-LCNN": PA2021_KEYS_ROOT / "CM" / "LFCC-LCNN" / "score.txt",
    "RawNet2": PA2021_KEYS_ROOT / "CM" / "RawNet2" / "score.txt",
}

# The official scored/leaderboard subset -- see PROJECT_PLAN.md section 3.2.
PA2021_REPORTED_PARTITION = "eval"

# --- Audio / DSP params ---
SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10

# MFCC (classical baseline, Phase 4.2)
N_MFCC = 20
MFCC_N_FFT = 512
MFCC_HOP_LENGTH = 160

# CQT (main system, Phase 4.3)
# NOTE: 96 bins (the number originally sketched in PROJECT_PLAN.md) violates the
# Nyquist constraint at 16kHz with the default fmin (~32.7Hz, C1) -- the top
# wavelet's own bandwidth pushes past 8kHz even though its center frequency
# (~7.9kHz) looks safe. Confirmed directly: 96 raises librosa's ParameterError,
# 90 does not. 90 bins = 7.5 octaves at 12 bins/octave, same fmin.
CQT_HOP_LENGTH = 256
CQT_N_BINS = 90
CQT_BINS_PER_OCTAVE = 12
CQT_FIXED_FRAMES = 400

# --- Training ---
RANDOM_SEED = 42
LCNN_BATCH_SIZE = 32  # keep modest given the 4GB VRAM GTX 1650
