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

# Results and checkpoints are grouped per phase, and within Phase 5 per model
# family, so a run's artifacts sit together instead of being distinguished only by
# filename prefix. Phase 6 keeps one directory per training run (its tag).
PHASE5_DIR = RESULTS_DIR / "phase5"
PHASE5_SVM_DIR = PHASE5_DIR / "svm"
PHASE5_RF_DIR = PHASE5_DIR / "rf"
PHASE6_DIR = RESULTS_DIR / "phase6"

PHASE5_MODELS_DIR = MODELS_DIR / "phase5"
PHASE6_MODELS_DIR = MODELS_DIR / "phase6"

for d in (MANIFESTS_DIR, SPLITS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
          PHASE5_DIR, PHASE5_SVM_DIR, PHASE5_RF_DIR, PHASE6_DIR,
          PHASE5_MODELS_DIR, PHASE6_MODELS_DIR):
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
#
# Set to 250 (4.0s), NOT the 400 originally sketched. Measured on the real cache:
# 2019 CQTs have a median of 267 frames and 90.8% are shorter than 400, while 2021
# eval projects to ~149 frames -- so 400 would pad 2019 by ~1.50x and 2021 by
# ~2.68x, a train/test mismatch of our own making on top of the real domain shift.
# It would also leave ~91% of training files receiving deterministic padding rather
# than the per-epoch random crops that variable-length caching exists to enable.
# 250 sits near the 2019 median so roughly half the training set is genuinely
# cropped. Swept over {150, 250, 400} in Phase 6.
CQT_FIXED_FRAMES = 250

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

# --- Phase 5 classical baseline (MFCC -> SVM / RF) ---
# RBF SVM scales ~O(n^2.2) here (benchmarked: 2k->0.1s, 5k->0.7s, 10k->6.2s,
# 20k->28s), so fitting the full 175,959-row train split would take ~56min for a
# SINGLE fit -- the implied Gram matrix alone is ~248GB at float64. The SVM is
# therefore fitted on a stratified subsample; the size is chosen empirically from
# a learning curve rather than guessed.
# Phase 5 runs a FULL FACTORIAL sweep: every subsample size x every (C, gamma).
# That is deliberately more compute than picking one size from a learning curve
# and grid-searching only there -- it yields a complete, uniform results table
# (no "best size chosen by one arbitrary hyperparameter setting" caveat) and lets
# the learning curve be read at each hyperparameter combination independently.
SVM_SWEEP_SIZES = [10_000, 20_000, 50_000, 80_000, 100_000, 150_000]
# Also sweep the entire train split. Kept as a flag rather than a literal size
# because the exact row count is data-dependent (currently 175,959) -- the code
# resolves it from the loaded split, so a future re-split can't silently make
# this constant wrong.
SVM_SWEEP_INCLUDE_FULL = True

# Per-size subsamples are drawn INDEPENDENTLY (each a fresh stratified draw from
# the full train split) rather than nested. Nesting removes a little between-size
# sampling noise, but it makes every size depend on the largest one -- so adding a
# new, larger size to the sweep would silently change all the smaller subsamples
# and invalidate every result already computed. Independent draws make the sweep
# extensible: previously-computed points stay exactly reproducible.

# Plateau rule: report the SMALLEST size whose dev EER is within this absolute
# margin of the best observed. 0.002 = 0.2 percentage points of EER.
SVM_PLATEAU_TOLERANCE = 0.002

# Log-spaced: both parameters act multiplicatively, so orders of magnitude are
# what matter. gamma="scale" = 1/(n_features * X.var()) ~= 1/120 = 0.0083 after
# standardisation, i.e. it sits between the 1e-3 and 1e-2 grid points.
SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]
SVM_GAMMA_GRID = ["scale", 1e-3, 1e-2, 1e-1]

# CORRECTION (made after the first Phase 5 run; see PROGRESS_REPORT.md).
# An earlier version of this file dropped columns 20-59 -- mean(delta) and
# mean(delta-delta) -- from the SVM's input, arguing that their tiny across-file
# variance (col-std 0.005-0.103, vs 3.5-46.0 for mean(MFCC)) meant StandardScaler
# would just amplify numerical noise into the RBF kernel's distance metric.
#
# That reasoning was wrong. Low ABSOLUTE variance is not low DISCRIMINATIVE power:
# a column with a tiny range still separates the classes well if bonafide and spoof
# sit at reliably different points inside that range. Two independent checks agree:
#   - the Phase 5 RF importance plot ranked mean(delta) the SECOND most important
#     of the six feature blocks, above three blocks that were never dropped;
#   - direct class-separation measurement: mfcc_24 (mean(delta), dropped) separates
#     bonafide from spoof by 0.63 pooled sigma, versus 0.24 for mfcc_60
#     (std(MFCC), kept).
# Standardisation is in fact the mechanism that makes this small-but-real signal
# usable by a distance-based kernel, rather than letting large-magnitude columns
# like mfcc_1 dominate every pairwise distance.
#
# Both the SVM and the Random Forest now use all 120 dimensions.

RF_N_ESTIMATORS = 300
RF_N_JOBS = 8

# --- Phase 6: packed CQT store ---
# Reading the 175,959 individual .npy files costs ~10.3 ms each COLD -- ~30 min per
# epoch of pure I/O, dwarfing the GPU work. Measured: mmap does not help (10.4 ms,
# i.e. the cost is per-file-open overhead, not data transfer), but random reads
# inside ONE already-open file run at ~0.27 ms -- a ~38x speedup, taking loading to
# under a minute per epoch. Hence the packing step.
PACKED_DIR = ASVSPOOF_ROOT / "packed"
PACKED_BLOB = {s: PACKED_DIR / f"cqt_{s}.dat" for s in ("train", "dev")}
PACKED_INDEX = {s: PACKED_DIR / f"cqt_{s}_index.parquet" for s in ("train", "dev")}

# --- Phase 6: LCNN ---
# Time-axis pooling only: collapses (32, 5, T') -> (32, 5, 1) = 160 features
# regardless of T. Two reasons over flattening:
#   1. it makes the T sweep interpretable -- with a flattened head the FC input
#      (and thus the parameter count) grows with T, so a larger T would mean a
#      bigger model, confounding "more context" with "more capacity";
#   2. it keeps the FREQUENCY axis intact, which is where the replay fingerprint
#      lives -- pooling over both axes would discard exactly that.
# "flatten" reproduces the original LCNN paper's head and is kept as an ablation.
LCNN_HEAD = "timepool"          # "timepool" | "flatten"
LCNN_DROPOUT = 0.7              # as in the LCNN paper
# Benchmarked on this GPU (ms/sample, fwd+bwd, AMP off):
#   batch  32 ->  2.60  peak 0.84 GB   ~7.6 min/epoch   <-- best
#   batch  64 ->  2.97  peak 2.32 GB   ~8.7 min/epoch
#   batch 128 -> 21.55  peak 4.61 GB  ~63.2 min/epoch
#   batch 256 -> 45.32  peak 6.10 GB ~132.9 min/epoch
# 128 and 256 exceed the card's 4.29 GB, so PyTorch spills into shared host memory
# -- which on a machine with only 5.9 GB system RAM is doubly punishing. Hence 32.
LCNN_BATCH_SIZE = 32
LCNN_EPOCHS = 30
LCNN_LR = 1e-3
LCNN_LR_FACTOR = 0.5            # ReduceLROnPlateau multiplier
LCNN_LR_PATIENCE = 3            # epochs without dev-EER improvement before decay
LCNN_EARLY_STOP_PATIENCE = 8    # epochs without dev-EER improvement before stopping
# AMP is DISABLED after measuring it. The GTX 1650's TU117 chip has no tensor
# cores, so FP16 buys no arithmetic speedup -- and the cast overhead makes it
# actively WORSE here: 227.8 ms/batch with AMP vs 82.3 ms without, a 2.8x
# slowdown. Its only real benefit is memory (peak 0.40 GB vs 0.84 GB), and memory
# is not the binding constraint at batch 32 (0.84 GB of ~3.6 GB usable). Revisit
# only if a future configuration actually runs out of VRAM.
LCNN_USE_AMP = False
# 0, i.e. load in the main process. This machine has only 5.9GB of system RAM, and
# DataLoader workers pass batches back through Windows shared-memory file mappings
# -- with 2 workers that failed outright with error 1455 (ERROR_COMMITMENT_LIMIT,
# "the paging file is too small"). The cost of going single-process is small and
# measured: loading is 10.7 ms/batch against 83 ms of GPU work, so ~11% per epoch.
LCNN_NUM_WORKERS = 0

# Input scaling. "unit" = x/255 (preserves everything, including absolute level).
# "cmvn" = per-utterance per-frequency-bin mean/variance normalisation. CMVN is
# standard in speech for removing channel effects -- but the replay fingerprint IS
# a channel effect, so it may erase the evidence. Run as a controlled ablation.
LCNN_NORM = "unit"              # "unit" | "cmvn"

# SpecAugment (applied to the cached CQT, training only -- essentially free).
# NOTE: frequency masking could mask exactly the high-frequency band this thesis
# argues carries the replay fingerprint. Treat these widths as dev-tuned, not as
# ASR defaults to be copied.
SPECAUG_N_TIME_MASKS = 2
SPECAUG_TIME_MASK_MAX = 25      # frames, out of CQT_FIXED_FRAMES
SPECAUG_N_FREQ_MASKS = 2
SPECAUG_FREQ_MASK_MAX = 8       # bins, out of CQT_N_BINS (90)

# --- Training ---
RANDOM_SEED = 42
