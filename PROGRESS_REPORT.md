# Progress Report — Phases 0 through 4

This is a narrative log of what's actually been done in the repo so far, with the
numbers that were verified and the bugs that came up along the way. `PROJECT_PLAN.md`
is the forward-looking plan; this file is the backward-looking record of execution
against it. Written to be read cold — includes reasoning, not just conclusions.

---

## Phase 0 — Environment setup

Installed into the shared venv (`OG/.venv`, Python 3.11): `librosa`, `scikit-learn`,
`soundfile`, `gradio`, `faster-whisper`, `sounddevice`, `seaborn`, and `pyarrow` (the
last one wasn't on the original planned list — it turned out to be required for
`pandas.to_parquet`/`read_parquet`, discovered when the first manifest-build run
crashed on `ImportError`).

Two pip installs in this session crashed with `UnicodeEncodeError` from pip's own
`rich`-based progress bar / logging trying to render the Cyrillic characters in the
project's path (`...Радна површина...`) under Windows' cp1252 console codepage. Both
times the crash was purely cosmetic — the packages had already installed successfully
before the crash (verified with `python -c "import X"` immediately after). Worth knowing
about since it'll likely recur: **a pip/print crash mentioning `charmap`/`cp1252` in this
repo almost certainly means the real work already finished — check before assuming
failure.**

Created the project skeleton: `manifests/`, `features/`, `splits/`, `models/`,
`results/`, `notebooks/`, `src/`, `src/demo/`.

## Phase 1 — Manifests (`src/manifest.py`)

Wrote `src/config.py` as the single source of truth for dataset paths (verified
directly against the on-disk layout under `E:\ASVspoof data\`) and DSP/training
hyperparameters.

`src/manifest.py` parses three raw protocol sources into clean, cached parquet tables
in `manifests/` (gitignored — regenerable, and they embed absolute local paths):

- **`pa2019_cm.parquet`** (218,430 rows) — one row per 2019 PA file with a CM label,
  built from `ASVspoof2019.PA.cm.{train,dev,eval}.{trn,trl}.txt`. Columns:
  `speaker_id`, `filename`, `env_id`, `attack_id`, `label`, `filepath`, `split`, `year`.
  Verified counts: train 5,400 bonafide / 48,600 spoof; dev 5,400 / 24,300; eval 18,090
  / 116,640.
- **`pa2019_asv_enroll.parquet`** (22,626 rows) — bonafide-only files sitting in the ASV
  enrollment `.trn` files (`ASVspoof2019.PA.asv.{dev,eval}.{female,male}.trn.txt`),
  exploded from their comma-separated file lists to one row each. These files are never
  referenced by the CM protocol, so they were otherwise invisible — a "free bonus" pool
  of extra bonafide data used in Phase 2. Split breakdown: 3,834 dev / 18,792 eval.
- **`pa2021_cm.parquet`** (943,110 rows) — the 2021 eval set's audio-side protocol
  lists filenames with no labels at all (by design, to prevent leaderboard peeking), so
  this manifest is the result of joining that filename list against the real labels in
  `PA-keys-full/keys/PA/CM/trial_metadata.txt`, and resolving each filename to its
  actual path via a glob across all 7 downloaded parts (`ASVspoof2021_PA_eval_part00`
  through `part06`). Partition breakdown: eval 721,332 / hidden 134,730 / progress
  87,048. Every filename resolved to exactly one file — no missing-path errors.

All three manifests' row counts and label distributions were verified by hand and match
the numbers anticipated in `PROJECT_PLAN.md` exactly.

## Phase 2 — Speaker-disjoint resplit (`src/resplit.py`)

Rationale (full detail in `PROJECT_PLAN.md` section 4): 2019's own eval split is folded
into the training pool entirely — this project doesn't need or use a 2019 eval split,
since 2021 PA eval is the sole true held-out test set, touched once at the very end.

`src/resplit.py`:
1. Pools `pa2019_cm.parquet` (218,430 rows) with `pa2019_asv_enroll.parquet` (22,626
   rows) into an **enriched pool of 241,056 rows, 107 speakers**, 51,516 bonafide /
   189,540 spoof — a **3.68:1** imbalance, down from the original CM-only 6.56:1, as
   anticipated.
2. Resplits that pool with `sklearn.model_selection.GroupShuffleSplit(groups=speaker_id,
   train_size=0.75, random_state=42)` — critically, the 75/25 split is over *speakers*,
   not rows, so no speaker appears in both train and dev. The script explicitly asserts
   zero speaker overlap and raises if that's ever violated.
3. Writes `splits/train_2019.csv` (175,959 rows, 80 speakers, 37,449 bonafide / 138,510
   spoof) and `splits/dev_2019.csv` (65,097 rows, 27 speakers, 14,067 bonafide / 51,030
   spoof).

The residual 3.68:1 imbalance is deliberately *not* further corrected by
over/undersampling — per the plan, it's handled downstream via `class_weight="balanced"`
(SVM/RF, Phase 5) and `pos_weight` (LCNN, Phase 6). Revisit only if dev EER later
suggests that's insufficient.

## Phase 3 — EDA (`src/eda.py`)

Generates everything into a new top-level `EDA/` folder — deliberately kept separate
from `results/`, which is reserved for actual model outputs (EER tables, DET curves,
etc.) from Phase 7 onward. `EDA/` is small (~1.1MB) and tracked in git as a lightweight
visual record; `results/` is gitignored.

Outputs:

1. **`01_class_balance_before_after.png`** — bonafide/spoof counts before vs. after the
   Phase 2 enrichment; annotates the 6.56:1 → 3.68:1 ratio change.
2. **`02_class_balance_train_dev.png`** — same, but for the actual `train_2019` /
   `dev_2019` files that downstream phases will train on.
3. **`03_duration_histograms.png`** + **`03_duration_summary.csv`** — duration
   distribution (via `soundfile.info()`, no full decode) for a stratified sample (1,500
   files per label) of the 2019 pool by label, and 2019 vs. 2021 eval. **Notable
   finding**: 2021 clip durations peak around ~2s, versus 2019's ~4s — a real
   distribution shift between the two datasets worth a mention alongside the
   domain-shift discussion in `PROJECT_PLAN.md` section 3.3.
4. **`04_waveform_spectrogram_cqt.png`** — for one speaker with both a bonafide and a
   (mild, attack-AA) spoof recording, shows waveform / STFT spectrogram / CQTgram
   side by side.
5. **`05_mfcc_vs_cqt.png`** — same pair, but MFCC vs. CQT, aimed directly at the
   thesis's central empirical argument (`PROJECT_PLAN.md` section 6): does the
   mel filterbank's compression of the high-frequency band destroy the
   loudspeaker/microphone replay fingerprint that CQT preserves? The MFCC panel is
   z-scored per-coefficient for display only (real feature extraction is untouched) —
   the raw values are dominated by the 0th coefficient (log-energy), which otherwise
   washes out the other 19 coefficients visually.
6. **`06_speaker_gender_balance.png`** — speaker counts per split, plus a gender
   breakdown derived from the ASV-enrollment `.trn` files (which are split into
   separate male/female files for dev and eval). Coverage is partial by construction:
   the original 20 PA train speakers have no ASV-enrollment record at all, so they show
   up as "unknown" rather than being guessed at.
7. **`07_attack_condition_distribution.png`** — bonus sanity check confirming all 9
   attack-ID combinations (attacker-distance × replay-device-quality) appear in both
   `train_2019` and `dev_2019`, in closely matched proportions — i.e. the resplit didn't
   accidentally skew condition coverage.

### Bugs found and fixed during Phase 3

- **CQT parameter bug (real, affects Phase 4 too)**: the plan's originally-sketched CQT
  config (96 bins, 12 bins/octave, default fmin ≈32.7Hz) is invalid at a 16kHz sample
  rate — `librosa.cqt` raises `ParameterError: Wavelet basis with max frequency=...
  would exceed the Nyquist frequency`. This is subtle: the top bin's *center* frequency
  (≈7.9kHz) looks safely under the 8kHz Nyquist limit, but the wavelet's own bandwidth
  pushes past it. Confirmed empirically that 96 bins fails and 90 bins doesn't (holding
  bins-per-octave and fmin fixed). **`config.CQT_N_BINS` is now 90, not 96** — this is
  the value Phase 4 feature extraction must use.
- **Washed-out MFCC visualization** (display-only, not a pipeline bug): raw MFCC values
  span a huge range dominated by the 0th coefficient, making a naive `specshow` look
  like a flat wall of one color. Fixed by z-scoring each coefficient independently
  before display in `05_mfcc_vs_cqt.png`.
- Same cosmetic Windows-console Unicode crash as in Phase 0 hit the very last
  `print()` statement in `eda.py` (after all figures were already saved to disk) —
  patched with `sys.stdout.reconfigure(encoding="utf-8")`.

---

## Phase 4 — DSP feature extraction (`src/features.py`)

### Dataset restructure (prerequisite, done before extraction)

Per instruction, the dataset directory was restructured on E: from `E:\ASVspoof
data\{ASVspoof2019_PA, ASVspoof2021_PA_eval, ...}` to a cleaner two-level layout:
`E:\ASVspoof\data\{...same contents...}` and a new sibling `E:\ASVspoof\features\`,
dedicated to this phase's cache. The local project's own (empty) `features/` folder
was deleted, and `config.py`'s `FEATURES_DIR` now points at the new E: location
instead of living under the project root. Rationale: the project directory sits
inside OneDrive sync scope, and dumping ~241,000 generated cache files there would be
slow to sync and pointless (fully reproducible); E: already hosts the raw dataset and
has more free space than C: to begin with.

Consequence that had to be handled: `manifests/*.parquet` and `splits/*.csv` all embed
**absolute** `filepath` columns pointing at the old `E:\ASVspoof data\...` location.
After the move those paths were stale, so `src/manifest.py` and `src/resplit.py` were
both re-run from scratch to regenerate them against the new `E:\ASVspoof\data\...`
paths. Row counts and the resplit's random assignment were identical to before (same
`random_state=42`), confirming nothing else was affected.

### A real, large-scale bug: ~46% of the corpus wouldn't decode

While benchmarking extraction speed, `soundfile.read()` started throwing `flac
decoder lost sync` / `unknown error in flac decoder` on a large fraction of files.
Quantified properly on a random sample of 500 files from the enriched pool: **232/500
(46.4%)** failed. This was not file corruption — `file` (the Unix utility) correctly
identified a failing file's header as a valid FLAC stream with the right sample
count/rate. The real cause: `soundfile`'s bundled `libsndfile` (1.2.2) has its own
internal FLAC decoder (not the reference `libFLAC`), and it can't handle whatever
encoding parameters a large chunk of this corpus was produced with. `librosa.load()`'s
automatic fallback to `audioread` didn't help either, since there was no `ffmpeg`
anywhere on this system for `audioread` to fall back to, and installing a from-source
alternative (`pyflac`) failed to build (no C compiler/CMake available).

**Fix**: installed `imageio-ffmpeg`, a pip package that bundles a portable static
ffmpeg binary (no admin rights or system install needed). Verified via direct
subprocess decode that ffmpeg reads the previously-failing file correctly, then
re-verified against fresh, larger samples with zero failures.

**Design decision**: decode *every* file uniformly through ffmpeg — not a
"try-soundfile-then-fall-back" hybrid. One consistent decode path for the whole
corpus is a stronger, simpler methods-section claim than mixing two decoders
depending on which one happened to work, and avoids any subtle numerical
inconsistency between them. Cost: ffmpeg subprocess decode is slower per file than
`soundfile` would have been (~76ms/file just for decode, benchmarked), but this
turned out to be entirely manageable (see timing below).

This fix is corpus-wide, not 2019-specific — `src/features.py`'s `load_audio()` will
be reused as-is for the 2021 streaming evaluation in Phase 7.

### Fixed-length CQT handling: resolved in favor of deferring to Phase 6

The plan sketches a fixed CQT frame count (~400 frames) for the LCNN, but also says
"random crop while training, center crop for eval" — those two are in tension if a
single crop is baked in once at cache time (there'd be nothing left to randomize
across epochs). Resolved by **caching each file's CQT at its natural, unpadded,
uncropped length now**, and deferring the pad/random-crop/center-crop-to-400 logic to
Phase 6's `Dataset` class, which will operate on these variable-length arrays and
genuinely re-crop on every epoch during training (real augmentation), while dev/eval
get a deterministic center crop (reproducible metrics).

Whether to cap the stored length (to bound the rare very-long outlier) was checked
empirically rather than assumed: a full header-only scan of all 241,056 files (not
just a sample) found duration `mean=4.53s, median=4.29s, max=17.5s`, with only
**1,804 files (0.748%) exceeding 10s** and just 117 (0.049%) exceeding 15s. Capping at
10s would have saved only ~17MB out of ~6.1GB total — negligible. **No cap was
applied**; every file is cached at its true natural length.

### `src/features.py`

Scope: only the 2019 pool (`train_2019.csv` + `dev_2019.csv`, 241,056 files combined).
2021 (943,110 files) is deliberately untouched here — its features get extracted
on-the-fly during Phase 7 evaluation and never cached, per the plan's disk-cost
reasoning (caching CQT for all 721,332 `partition=="eval"` files would cost ~27GB for
a set that's only scored once).

Two outputs:
- **MFCC**: 20 MFCC + delta + delta² (60×T) → mean+std pooled to a fixed **120-dim
  vector** per file, exactly as sketched in the plan. Cached as **one consolidated
  parquet** (`E:\ASVspoof\features\mfcc\pooled_mfcc.parquet`) rather than 241,056 tiny
  files, since the vectors are tiny (~480 bytes each) and a classical model wants a
  single feature table anyway.
- **CQT**: log-power CQTgram at 90 bins (see Phase 3's Nyquist-bug note), quantized to
  **uint8** via `librosa.amplitude_to_db(ref=max, top_db=80)` linearly mapped to
  [0, 255] — the same convention already used for the EDA plots. Cached as **one
  `.npy` per file** (`E:\ASVspoof\features\cqt\<filename>.npy`), since ragged
  variable-length arrays can't stack into one dense file.

Engineering details worth recording:
- **Resumable by design, and this mattered in practice**: the script checkpoints the
  MFCC parquet after every 5,000-file chunk, and on startup skips any file whose MFCC
  row is already checkpointed *and* whose CQT `.npy` already exists. This session hit
  two unrelated environment crashes while this phase was in progress (once right after
  the E: drive restructure, once mid-way through writing `features.py`), and the
  resumable design meant neither crash cost any lost extraction work — a coincidental
  but real validation of the design choice, not just defensive theatre.
- Benchmarked before committing to the full run: **~155ms/file** single-threaded
  (ffmpeg decode + MFCC + CQT combined). Parallelized via `joblib` across 8 of the
  machine's 12 logical cores (`config.FEATURE_EXTRACTION_N_JOBS`).
- Failures (if any) get logged to `extraction_failures.csv` and skipped rather than
  crashing the whole run — moot in the event, since there were none.

### Verified results

Full run completed cleanly: **241,056 / 241,056 files processed, 0 failures**
(no `extraction_failures.csv` was even created) — the ffmpeg decode fix held at full
corpus scale, not just in samples.

- `E:\ASVspoof\features\cqt\`: 241,056 `.npy` files, **6.3GB** total (in line with the
  ~6.1GB estimate). Spot-checked: `(90, T)` uint8 arrays with genuinely varying `T`
  per file (e.g. 284/268/278 frames across a random sample) and values spanning the
  full 0–255 range.
- `E:\ASVspoof\features\mfcc\pooled_mfcc.parquet`: shape `(241056, 127)` — 7 identifier
  columns (`filename`, `speaker_id`, `label`, `subset`, `env_id`, `attack_id`, `split`)
  + 120 MFCC feature columns, **171MB**, zero NaNs.
- Row counts match Phase 2 exactly: `subset` 175,959 train / 65,097 dev; `label`
  189,540 spoof / 51,516 bonafide.

---

Next: **Phase 5**, the classical MFCC→SVM/RF baseline, per `PROJECT_PLAN.md` section 6.
