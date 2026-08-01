"""Single source of truth for paths and hyperparameters."""
from pathlib import Path

# --- Roots ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASVSPOOF_ROOT = Path("E:/ASVspoof")
DATA_ROOT = ASVSPOOF_ROOT / "data"

# Feature cache lives on E: (not under PROJECT_ROOT) for two reasons: the project
# directory is inside OneDrive sync scope, and syncing hundreds of thousands of
# generated cache files would be slow and pointless; and C: has much less free
# space than E:, which already hosts the raw dataset. See PROGRESS_REPORT.md.
FEATURES_DIR = ASVSPOOF_ROOT / "features"

MANIFESTS_DIR = PROJECT_ROOT / "manifests"
SPLITS_DIR = PROJECT_ROOT / "splits"
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
# Phase 4 caches each file's CQT at its own natural length, uncapped (verified
# across all 241,056 files: only 0.75% exceed 10s, and capping saves a negligible
# ~17MB out of ~6.1GB total -- not worth the extra parameter/edge case). This
# constant is consumed only in Phase 6's Dataset, which pads short clips and
# randomly (train) or centrally (dev/eval) crops long ones down to this window.
CQT_FIXED_FRAMES = 400

# --- Phase 4 feature extraction ---
# CQT dB -> uint8 quantization floor (matches librosa.amplitude_to_db's top_db
# default): values are linearly mapped from [-CQT_TOP_DB, 0] dB to [0, 255].
CQT_TOP_DB = 80.0
# ~46% of the 2019 corpus fails to decode via soundfile/libsndfile ("flac decoder
# lost sync") -- confirmed at scale (232/500 sampled), root-caused to a libsndfile
# FLAC-decoder limitation, not file corruption (raw headers check out fine).
# Every file is decoded via ffmpeg (through the portable imageio-ffmpeg binary)
# instead, uniformly, for one consistent decode path across the whole corpus.
FEATURE_EXTRACTION_N_JOBS = 8  # leave a few of the 12 logical cores free

# --- Training ---
RANDOM_SEED = 42
LCNN_BATCH_SIZE = 32  # keep modest given the 4GB VRAM GTX 1650
