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

# --- 9.8c in-house cepstral-GMM fusion partners --------------------------------
# LFCC, standard ASVspoof recipe. The ONE thing separating this from MFCC is that
# the filterbank is spaced LINEARLY: mel spacing compresses the top octaves, which
# is exactly where the replay fingerprint lives, and is why LFCC beats MFCC on PA.
LFCC_N_FFT = 512
LFCC_WIN_LENGTH = 320           # 20 ms at 16 kHz
LFCC_HOP_LENGTH = 160           # 10 ms
LFCC_N_FILTERS = 70
LFCC_N_COEFF = 20               # x {static, delta, delta-delta} = 60 dims

# Constant-Q cepstral features read back out of the Phase 4 uint8 CQT cache.
# NOT CQCC and must never be called that (PROJECT_PLAN 9.8c): real CQCC resamples
# the geometrically-spaced bins onto a uniform scale before the DCT, which this
# cache cannot support. Also: C0 carries no absolute level, because the cache is
# peak-normalised per file (ref=np.max), and quantisation is CQT_TOP_DB/255 dB.
CQTDCT_N_COEFF = 20             # likewise x3 = 60 dims

# GMM back-end, matching the official baselines' component count. Fitted by
# chunked exact EM (src/gmm.py) rather than sklearn, whose (n_frames x k) float64
# responsibility matrix does not fit here -- see that module's docstring.
GMM_N_COMPONENTS = 512
GMM_MAX_ITER = 30
GMM_TOL = 1e-4                  # on mean per-frame log-likelihood
GMM_REG_COVAR = 1e-6
GMM_CHUNK = 20_000              # frames per E-step chunk: 20k x 512 x 8B = 82 MB
# Frames sampled per file, set PER CLASS to reach a comparable total, because the
# split is ~1:9 bonafide:spoof. Equal frames per file *within* a class is what
# matches the scoring rule (mean frame log-likelihood, so every file counts once)
# and avoids the duration bias 6.10 measured: bonafide files are systematically
# longer than spoof (323 vs 274 frames), and duration alone scores 41.5% EER.
GMM_TARGET_FRAMES_PER_CLASS = 1_800_000
GMM_MIN_FRAMES_PER_FILE = 4
# Sampled training frames, fitted models and per-file LLR score tables. On E: with
# every other bulky intermediate; the frame stores are ~430 MB per class per
# feature, and NO frame-level store is ever written for 2021 (that would be ~79 GB
# -- scoring streams instead).
GMM_DIR = ASVSPOOF_ROOT / "gmm"
# In-house partners. A FOURTH registry, kept separate for the same reason as
# PHASE7_FUSION_SYSTEMS: these are ours, post-hoc, and unlike everything in
# PHASE7_LCNN_SYSTEMS they are GMMs rather than LCNNs, with no checkpoint to load.
PHASE7_INHOUSE_GMM_SYSTEMS = {
    "our-LFCC-GMM":    "lfcc",
    "our-CQT-DCT-GMM": "cqtdct",
}
PHASE7_INHOUSE_GMM_BY_FEAT = {v: k for k, v in PHASE7_INHOUSE_GMM_SYSTEMS.items()}

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

# --- Phase 7: evaluation on the held-out 2021 PA eval set ---
# PRE-REGISTERED SYSTEMS, fixed before 2021 was touched (PROJECT_PLAN.md phase 7).
# Ordered best-to-worst by dev EER; the dev numbers are recorded here purely so the
# registration is legible in one place -- they are NOT read by any code.
#
# The list extends PROJECT_PLAN.md's original five LCNNs by two, both trained in
# Phase 6 and both declared before any 2021 score existed:
#   T400  -- the MATCHED CONTROL that makes two of the three registered predictions
#            testable at all. `cmvn_T400` and `baseline_T250` are timepool models
#            while the primary `flatten_T400` is flatten, so comparing either
#            against the primary varies TWO things at once (norm+head, or T+head).
#            T400 is timepool/unit, so cmvn_T400 differs from it only in norm and
#            baseline_T250 only in T. Without it predictions 1 and 3 are confounded.
#   T150  -- completes the T axis (150/250/400, all timepool) for ~5 min of GPU.
PHASE7_LCNN_SYSTEMS = {
    "flatten_T400": 0.00798,        # primary -- best on dev
    "T400": 0.00902,                # matched control (timepool, unit)
    "cmvn_T400": 0.01293,           # prediction 3: CMVN may transfer better
    "flatten_T400_aug1": 0.01486,   # mild waveform augmentation (50% clean)
    "flatten_T400_aug": 0.02353,    # aggressive waveform augmentation (25% clean)
    "baseline_T250": 0.02780,       # prediction 1: less 2019->2021 padding mismatch
    "T150": 0.06584,                # completes the T axis
}
# Classical Phase 5 baselines, scored from the cached 2021 MFCC in a second pass.
PHASE7_CLASSICAL_SYSTEMS = {"MFCC-SVM": 0.09216, "MFCC-RF": 0.11736}

# --- POST-HOC systems (NOT pre-registered) -------------------------------------
# Trained after 2021 results were seen, so they carry none of the guarantee the
# dicts above do. Deliberately a SEPARATE registry: the Phase 7 table must remain
# exactly as pre-registered, and post-hoc work is reported in its own section.
#
# The value is the partitions each system may be scored on, enforcing the decision
# rule declared in PROJECT_PLAN.md 9.3.1 *in code* rather than by memory:
#   - both candidates were compared on `progress` (the declared decision set);
#   - ONLY the winner (timepool) was taken to `eval`, once;
#   - the loser (flatten) is never scored on eval, so no post-hoc "best of two on
#     eval" can be reported even by accident.
# `hidden` is neither, so both may be scored there (it feeds the 7.19 analysis).
PHASE7_POSTHOC_SYSTEMS = {
    "timepool_T150_aug": ("progress", "eval", "hidden"),   # winner
    "flatten_T150_aug":  ("progress", "hidden"),           # not selected
    # --- 9.1 dose sweep. Same T150+timepool configuration as the incumbent above,
    # varying ONLY p(clean): the probability of drawing the unaugmented blob. No new
    # copies are generated -- these reweight cqt_train_aug{1,2,3}, which is what
    # separates dose from diversity (9.1.1). `timepool_T150_aug` IS the p=0.25 point.
    # All four are `progress`-only until the declared rule names a winner, so none can
    # reach eval even by accident.
    "timepool_T150_pc50": ("progress",),    # p(clean)=0.500, equivalent to 1 copy
    "timepool_T150_pc17": ("progress",),    # p(clean)=0.167, equivalent to 5 copies
    "timepool_T150_pc12": ("progress",),    # p(clean)=0.125, equivalent to 7 copies
    "timepool_T150_pc06": ("progress",),    # p(clean)=0.0625, equivalent to 15 copies
    # --- 9.2 T sweep below 150. Same timepool head and the incumbent's exact
    # augmentation (3 copies, uniform draw => p(clean)=0.25), so T is the ONLY variable
    # that moves against `timepool_T150_aug`. `progress`-only until one is declared.
    "timepool_T100_aug": ("progress",),
    "timepool_T75_aug":  ("progress",),
}
# The dose each tag was trained at, so score_posthoc/report cannot drift from the runs.
PHASE7_DOSE_SWEEP = {
    "timepool_T150_pc50": 0.5,     "timepool_T150_aug":  0.25,
    "timepool_T150_pc17": 0.167,   "timepool_T150_pc12": 0.125,
    "timepool_T150_pc06": 0.0625,
}

# (FUSED systems are registered further down, next to their score files --
#  see PHASE7_FUSION_SYSTEMS after PA2021_FUSION_SCORES.)

# Which partitions to extract. The HEADLINE number is PA2021_REPORTED_PARTITION
# ("eval", 721,332 rows) exactly as pre-registered; `progress` and `hidden` are
# extracted in the same pass (+~30% runtime) purely as a free consistency check,
# and the four official baseline score.txt files already cover all three.
PHASE7_PARTITIONS = ("eval", "progress", "hidden")

PHASE7_DIR = RESULTS_DIR / "phase7"
# Phase 7 results are split in two, and the split is methodological rather than
# cosmetic:
#
#   preregistered/  The Phase 7 deliverable, FROZEN. Produced by the single scored
#                   pass over the held-out set against the system list fixed in
#                   advance (section 6, phase 7). Nothing is added here again.
#                   Re-running report_2021 reproduces it byte-for-byte, since the
#                   inputs and code are deterministic -- but no NEW artifact belongs
#                   in it, because nothing designed after 2021 was seen can carry
#                   the guarantee the contents of this folder carry.
#
#   posthoc/        Everything since, and everything from now on. Designed with
#                   knowledge of the 2021 results, so it carries no pre-registration
#                   guarantee and must never be tabulated as though it does.
#
# The bootstrap CIs live in posthoc/ even though they describe pre-registered
# systems: the ANALYSIS was designed after the fact, and the frozen folder holds the
# deliverable as it stood, not every later thing said about it.
PHASE7_PREREG_DIR = PHASE7_DIR / "preregistered"
PHASE7_POSTHOC_DIR = PHASE7_DIR / "posthoc"
# score.txt exports, one per system, in the official ASVspoof submission format
# (`FILENAME SCORE`, higher = bonafide -- the convention metrics.py fixes
# project-wide). ~20MB each, so gitignored; the summaries/figures beside them are
# tracked. Exporting these lets an examiner recompute every reported EER from a
# text file, without the 45GB corpus or this pipeline.
# Split to match: each folder's score.txt exports cover exactly its own systems, so
# the pre-registered set cannot be silently widened by a stray export.
PHASE7_PREREG_SCORES_DIR = PHASE7_PREREG_DIR / "scores"
PHASE7_POSTHOC_SCORES_DIR = PHASE7_POSTHOC_DIR / "scores"

# All bulky Phase 7 intermediates live on E:, never in the OneDrive-synced repo.
PA2021_WORK_DIR = ASVSPOOF_ROOT / "phase7_2021"
PA2021_CQT_SHARD_DIR = PA2021_WORK_DIR / "cqt"
PA2021_MFCC_SHARD_DIR = PA2021_WORK_DIR / "mfcc"
PA2021_SCORE_SHARD_DIR = PA2021_WORK_DIR / "lcnn_scores"
PA2021_CLASSICAL_SHARD_DIR = PA2021_WORK_DIR / "classical_scores"
PA2021_CQT_INDEX = PA2021_WORK_DIR / "cqt_index.parquet"
PA2021_LCNN_SCORES = PA2021_WORK_DIR / "lcnn_scores.parquet"
PA2021_CLASSICAL_SCORES = PA2021_WORK_DIR / "classical_scores.parquet"
# Post-hoc system scores, kept in their own file so the pre-registered score table
# cannot be silently widened -- see PHASE7_POSTHOC_SYSTEMS above.
PA2021_POSTHOC_SCORES = PA2021_WORK_DIR / "posthoc_scores.parquet"
# Fused scores from the single eval application of PROJECT_PLAN 9.8b.1. A THIRD file
# for a third kind of object -- see PHASE7_FUSION_SYSTEMS.
PA2021_FUSION_SCORES = PA2021_WORK_DIR / "fusion_eval_scores.parquet"

# --- FUSED systems (post-hoc) ----------------------------------------------------
# A separate registry, because a fusion is a different kind of object from the two
# single-system dicts above, and the properties that separate it are the easiest to
# lose track of. Each entry maps a display tag to (score file, column in that file).
#
#   fusion_ours+2GMM          9.8b. NOT zero-shot (weights fitted on 87,048 labelled
#                             2021 trials) and CONTAINS two official baselines, so it
#                             can never be compared against them: you cannot beat a
#                             baseline by including it (9.8b.1a.1).
#   inhouse_fusion_progress   9.8c. Every component ours, so the baseline comparison
#                             is legitimate -- but still not zero-shot.
#   inhouse_fusion_dev        9.8c. Every component ours AND weights fitted on 2019
#                             dev, so it never sees a 2021 LABEL. It does read
#                             unlabelled target scores for z-normalisation (9.8c.2),
#                             which makes it transductive but not label-dependent.
#
# All three stay OUT of the CI-width aggregate in bootstrap_ci: that statistic backs a
# published claim about the 14 zero-shot single systems, and a fusion is not one.
# Kept out of PHASE7_POSTHOC_SYSTEMS for a practical reason too -- score_posthoc.py
# iterates that registry to run LCNN forward passes, and a fusion has no checkpoint.
PHASE7_FUSION_SYSTEMS = {
    "fusion_ours+2GMM":        (PA2021_FUSION_SCORES, "ours+2GMM"),
    "inhouse_fusion_dev":      (PA2021_WORK_DIR / "inhouse_fusion_dev_eval.parquet",
                                "inhouse_dev"),
    "inhouse_fusion_progress": (PA2021_WORK_DIR / "inhouse_fusion_progress_eval.parquet",
                                "inhouse_progress"),
}
PA2021_FAILURES_CSV = PA2021_WORK_DIR / "extraction_failures.csv"

# The 2021 pooled MFCC table IS cached (~500MB for all partitions), unlike the
# 27GB-at-the-time CQT estimate that PROJECT_PLAN.md 4.4 rejected. It is 0.05% of
# that, and it decouples classical scoring from the extraction pass -- libsvm would
# otherwise fight the 8 extraction workers for cores. It also makes the classical
# scores re-runnable, and would let a future model be scored with no re-extraction.

# PROJECT_PLAN.md 4.4 rejected caching 2021 CQT at an estimated ~27GB -- but that
# assumed 96 bins x 400 PADDED frames. Neither holds: 90 bins (the Nyquist fix),
# cached at NATURAL length as in Phase 4. Measured mean is 149.4 frames, so
# 90 x 149.4 x 943,110 = ~12.7GB, against 31.7GB free on E:.
#
# Worth it as insurance rather than for speed: the dominant Phase 7 risk is finding
# a bug AFTER a ~4h pass (wrong crop, normalisation mismatch, model loaded at the
# wrong T). Cached, re-scoring all nine systems is a ~1.5h GPU job with no CPU work;
# uncached it is another full re-extraction. Writing it is nearly free -- the arrays
# already pass through the main process on their way to the GPU.
#
# Stored as ONE BLOB SHARD PER CHUNK, not one monolithic blob: merging shards at the
# end would mean ~25GB of I/O and a transient 25.4GB footprint against 31.7GB free.
# The index therefore carries a `shard` column alongside `offset`/`n_frames`.
# Per-file .npy is not an option -- Phase 6 measured 10.3 ms per open, which over
# 943,110 files is 2.7h just to read the cache back once (see PACKED_DIR above).
PHASE7_CACHE_CQT = True

# Files per chunk. Sized so one chunk's returned arrays stay small in a 5.9GB
# machine: 4,000 x 90 x ~149 bytes = ~54MB of CQT plus ~2MB of MFCC.
PHASE7_CHUNK_SIZE = 4000
# Forward-only, so activations are not retained; 64 x 90 x 400 x 4B = 9.2MB/batch.
PHASE7_EVAL_BATCH = 64
# Extract chunk i+1 on a background thread while the GPU scores chunk i. The GPU
# work (~1.5h) does not otherwise overlap the CPU extraction (~2.4-2.9h), so this
# saves ~25%. Costs one extra chunk in flight (~56MB). --no-prefetch disables it.
PHASE7_PREFETCH = True
# Random files re-extracted from source and compared BYTE-FOR-BYTE against the
# packed blob. Same check pack_features.py runs, for the same reason: a silent
# offset bug would corrupt every score with nothing surfacing as an error.
PHASE7_VERIFY_N = 200
# Rebuild the joblib/loky worker pool every N chunks. loky reuses one pool across
# every Parallel() call, so over hundreds of chunks those 8 long-lived interpreters
# accumulate heap fragmentation and OS handles. Measured the hard way: a run died
# after 117 chunks (468,000 files, 1.5h) with WinError 1450
# (ERROR_NO_SYSTEM_RESOURCES) raised inside loky's result transport while returning
# a ~54MB chunk -- not a per-file error, so it escaped the failure guard and killed
# the run outright. 20 chunks = 80,000 files between rebuilds, a few seconds each.
PHASE7_RECYCLE_EVERY = 20

for d in (PHASE7_DIR, PHASE7_PREREG_DIR, PHASE7_POSTHOC_DIR,
          PHASE7_PREREG_SCORES_DIR, PHASE7_POSTHOC_SCORES_DIR,
          GMM_DIR, PA2021_WORK_DIR, PA2021_CQT_SHARD_DIR,
          PA2021_MFCC_SHARD_DIR, PA2021_SCORE_SHARD_DIR, PA2021_CLASSICAL_SHARD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Training ---
RANDOM_SEED = 42
