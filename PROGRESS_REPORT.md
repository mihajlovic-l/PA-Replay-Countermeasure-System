# Progress Report — Phases 0 through 6

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

## Phase 5 — Classical MFCC baseline (`src/metrics.py`, `src/train_classical.py`)

### `src/metrics.py` and the project-wide score convention

Written in Phase 5 because tuning needs EER, but deliberately shared with Phase 7
rather than duplicated. It fixes one convention for the whole project: **higher
score = more bonafide**, with **1 = bonafide (positive class), 0 = spoof**. This
matches the orientation of the official ASVspoof baseline `score.txt` files
(log-likelihood ratios favouring bonafide), so Phase 7 can join this project's
scores against CQCC-GMM / LFCC-GMM / LFCC-LCNN / RawNet2 with no sign flipping —
removing a whole class of silent sign-error bugs.

Contains `compute_eer`, `eer_from_labels`, `metrics_at_threshold` (confusion
matrix + precision/recall/F1/accuracy/ROC-AUC at one operating point) and
`full_report`.

**A correction to the plan recorded here**: `PROJECT_PLAN.md` section 5 specifies
`SVC(probability=True)` because "probability output needed for EER". That is not
right. EER is read off the ROC curve and is therefore purely **rank-based** — any
monotonically-ordered score works, and `decision_function` supplies one.
`probability=True` triggers an internal 5-fold Platt-scaling cross-validation at
roughly 5x the fit cost, and since Platt scaling is monotonic it cannot change the
EER anyway. The implementation uses `decision_function` throughout.

### A methodological mistake, and its correction

The first Phase 5 run dropped 40 of the 120 pooled-MFCC dimensions — columns 20-59,
`mean(delta)` and `mean(delta-delta)` — from the SVM's input. The stated reasoning
was that their across-file variance is tiny (col-std 0.005-0.103, versus 3.5-46.0
for `mean(MFCC)`) because the time-average of a derivative telescopes toward zero,
so `StandardScaler` would rescale numerical noise up to unit variance and pollute
the RBF kernel's distance metric.

**That reasoning was wrong: low absolute variance is not low discriminative power.**
A column with a tiny range still separates the classes well if bonafide and spoof
sit at reliably different points inside that range. Two independent checks agree:

- the Random Forest importance plot (which always saw all 120 features) ranked
  `mean(delta)` the **second most important of the six feature blocks**, above
  three blocks that were never dropped;
- direct class-separation measurement: `mfcc_24` (`mean(delta)`, dropped) separates
  bonafide from spoof by **0.63 pooled sigma**, versus **0.24** for `mfcc_60`
  (`std(MFCC)`, kept).

Standardisation is in fact the mechanism that makes such small-magnitude signal
*usable* by a distance-based kernel, instead of letting large-magnitude columns
like `mfcc_1` dominate every pairwise distance.

**Cost of the error**: the SVM improved from 11.809% to 9.896% dev EER at the same
50,000-sample training size once the features were restored — **1.9 pp, ~16%
relative**. It also inverted a conclusion: before the fix the Random Forest
(11.736%) appeared to beat the SVM (11.809%), and that finding would have been
written up. Afterwards the SVM leads by a wide and widening margin.

The general lesson, recorded because it applies to the rest of the project:
feature-importance and variance statistics are for **understanding**, not for
deciding what to feed a model. The same caution applies to permutation importance
if it is added later.

### Design: a full factorial sweep

The original design was "learning curve at one fixed hyperparameter setting, pick a
size, then grid-search only at that size". It was replaced with a **full factorial
sweep of every subsample size x every (C, gamma)** — 7 x 4 x 4 = **112 SVM fits** —
plus a Random Forest at each of the same sizes and on the full train split.

This costs far more compute but removes a real caveat: under the old design the
chosen training size depended on one arbitrary hyperparameter setting. The uniform
table also answers a question the old design structurally could not — whether the
optimal `C`/`gamma` themselves shift with data quantity (they do not; see below).

Why an RBF SVM has to be subsampled at all: cost scales roughly O(n^2.2) here
(benchmarked 2k->0.1s, 5k->0.7s, 10k->6.2s, 20k->28s), and the implied Gram matrix
at full train size is ~248GB in float64. It is never materialised — libsvm
recomputes chunks against a bounded cache — which is exactly why it degrades so
sharply with n.

Tuning is done directly on the **speaker-disjoint 2019 dev split, not by
cross-validation on train**: CV folds would share speakers, making them a strictly
worse generalisation estimate than the honest dev split Phase 2 already built.

**Per-size subsamples are independent stratified draws, not nested.** Nesting
removes a little between-size sampling noise, but it makes every size depend on the
largest one — so adding 80k/100k/150k/full to the sweep would silently have changed
the 50k subsample and invalidated the 16 points already computed at that size. This
was verified concretely before switching: under independent drawing the 50k indices
come out **bit-identical** to what the earlier run used (so those points were
genuinely reusable, saving ~2.4h), while under nesting they would not have.

Work is ordered **cheapest-first** — sizes ascending, and within each size the very
slow `gamma=0.1` points last — so an early interrupt still leaves every useful
`(C, gamma)` pair computed. This mattered in practice: `gamma=0.1` consumed **72.8%
of the total fitting time** while producing the worst results in the sweep.

Everything is resumable at single-point granularity (the sweep CSV is written after
every fit; RF checkpoints are keyed by size). Over the ~22h + ~15h of sweeping this
was exercised repeatedly, including one crash that killed a run after a single
completed point — nothing was ever recomputed unnecessarily.

### Results

**Best: n_train=175,959 (the entire train split), C=1.0, gamma=0.01 → 9.216% dev
EER** (37,943 support vectors, 21.6% of training data).

Best SVM per size, against the Random Forest at the same sizes:

| n_train | SVM | RF | gap |
|---|---|---|---|
| 10,000 | 11.395% | 12.929% | +1.53 pp |
| 20,000 | 10.628% | 12.623% | +2.00 pp |
| 50,000 | 9.896% | 12.155% | +2.26 pp |
| 80,000 | 9.618% | 12.057% | +2.44 pp |
| 100,000 | 9.423% | 11.856% | +2.43 pp |
| 150,000 | 9.363% | 11.752% | +2.39 pp |
| 175,959 (full) | **9.216%** | 11.736% | +2.52 pp |

Fitting RF at every SVM size (not only at full train) is what makes this comparison
honest — without size-matched rows, an RF result could be attributed to a larger
training set rather than to the algorithm.

**Hyperparameter sensitivity — the two parameters behave oppositely:**

- **`C` is sharply peaked but correctly located.** `C=1.0` won at every single size.
  Its neighbours are ~1 pp worse and nearly symmetric (at 100k: `C=0.1` -> 10.434%,
  `C=10` -> 10.312%), which is the clearest evidence the optimum sits where we
  sampled. Fitting a parabola in log10(C) through the three points around each
  minimum puts the vertex at C≈0.78-1.23 for all large sizes, with a predicted gain
  over `C=1` of **~0.01 pp**. Not worth refining.
- **`gamma` is flat near its optimum.** `scale` (= 1/(n_features x X.var()) = 1/120
  = 0.00833 after standardisation) and `0.01` differ by 0.03-0.05 pp, and which of
  the two wins flips between sizes — noise. The honest statement is **"the optimum
  is gamma ≈ 0.008-0.01 and the exact value within that range is immaterial"**,
  not that a particular value won.

**`gamma=0.1` failure, explained mechanically.** It scored 17.7-23.0% EER across all
C, took 10x longer to fit, and produced ~46,600 support vectors out of 50,000
training samples. Measuring actual kernel similarities on the data shows why: at
`gamma=0.1`, **99.9% of point pairs have K < 0.01**. No training example can inform
the prediction for any other, so the model has no option but to memorise each one
individually. At `gamma=0.001` the opposite happens (mean K = 0.80, everything looks
alike, boundary too smooth). `scale`/`0.01` sit in the well-conditioned middle —
which is exactly what the `scale` heuristic is engineered to produce: it makes
`gamma * E[||xi-xj||^2] ≈ 2`, verified on our data as 0.00833 x 236.8 = 1.97.

**One documented, deliberate non-pursuit**: at n=100k the `gamma=0.001` row peaks at
`C=100`, the edge of the searched grid (12.98 -> 11.26 -> 9.97 -> 9.93 across C).
This is the classic C-gamma compensation — a smoother kernel needs weaker
regularisation, so the good region runs along a diagonal ridge. It was not extended,
because that row is plateauing (the C=10 -> C=100 gain is only 0.04 pp) toward ~9.9%,
which is still worse than the 9.42% reached at `gamma=scale`. Even a perfect `C`
there could not reach the global optimum.

**Random Forest tree count, checked rather than assumed.** `n_estimators=300` was
chosen by convention. Evaluating the saved full-train forest with truncated
sub-forests shows dev EER plateaus at **~100 trees** (11.916%, versus 11.736% at
300), and 150 trees is *marginally worse* than 100 — non-monotonicity that confirms
we are past the plateau and into the noise floor. 300 is therefore validated as
"safely past the plateau" but is ~3x larger than necessary: 483MB and ~152s to fit,
versus ~160MB and ~50s at 100 trees, to buy 0.18 pp. Left at 300 since it is already
computed and genuinely the best value, but 150 is the better setting if RF reappears
in Phase 7 or the Phase 9 demo, where a 483MB model is inconvenient.
(`results/phase5_rf_n_estimators.{csv,png}`.)

Gini (`criterion="gini"`, sklearn's default, never set explicitly) is retained for
`feature_importances_`. Note the known MDI caveats — bias toward high-cardinality
features, and unpredictable credit-splitting among correlated features, which does
apply here since delta/delta-delta are derived from the MFCCs. This is why the
dropped-feature correction above was confirmed with an independent class-separation
measurement rather than resting on the importance plot alone.

### Bugs found and fixed during Phase 5

- **`gamma` dtype inconsistency corrupting grouped plots.** Rows loaded from CSV on
  a resumed run carried `gamma` as strings while freshly-computed rows held Python
  floats. `groupby(["C","gamma"])` then treated `"0.001"` and `0.001` as different
  groups, leaving the migrated 50k points as unconnected singletons scattered off
  their curves. Only the numeric gammas were affected (`"scale"` was a string
  either way), which is why some 50k points still sat correctly. Fixed at the source
  by normalising `gamma` to `str` on both read and write. The saved CSV was always
  correct — only the in-memory frame was mixed — so no recomputation was needed.
- **Stale size-matched RF checkpoint (latent).** The RF fitted at the chosen size was
  saved to a fixed `rf_mfcc_sub.joblib`. Had the chosen size ever changed, the resume
  logic would have loaded that file and reported a model trained at one size under
  another size's label. Fixed by keying the filename on `n_train`.
- **Heatmap colour scale inverted.** The `C x gamma` heatmap used `viridis_r`, which
  renders *low* EER bright and *high* EER dark — the opposite of the intuitive
  reading. Corrected to plain `viridis` (darker = lower EER = better), with the text
  contrast logic flipped to match and gamma columns ordered as configured rather than
  alphabetically.
- The 16-entry legend in the per-combination panel overlapped the data; moved below
  the axes.

### Measurements from this phase that constrain later phases

These are findings, not plans — the corresponding decisions are recorded in
`PROJECT_PLAN.md`.

- **The SVM learning curve never plateaued.** The final step (150k -> 175,959) still
  gained 0.147 pp, and there is no more 2019 data to add. A 120-feature RBF SVM is
  still data-hungry at the ceiling of what exists; an LCNN with far more parameters,
  trained on the same exhausted pool, will be more so.
- **`CQT_FIXED_FRAMES = 400` (6.40s) is longer than most files in either dataset.**
  Measured on the actual Phase 4 cache: 2019 CQTs have median **267** frames and
  **90.8% are shorter than 400**. 2021 eval projects to ~149 frames at its 2.39s mean,
  with even its longest file (7.03s) barely exceeding the window. So 2019 would be
  padded ~1.50x and 2021 ~2.68x — a train/test mismatch introduced by our own
  windowing, on top of the real domain shift. It also defeats the reason
  variable-length CQT was cached in the first place: random cropping only engages
  when a file is *longer* than the window, so at T=400 roughly 91% of training files
  would receive deterministic padding rather than per-epoch random crops.
- **2021's pooled MFCC statistics are inherently ~1.38x noisier than 2019's.** Mean/std
  pooling averages over ~453 frames on 2019 versus ~239 on 2021, and the standard
  error of a pooled statistic scales as 1/sqrt(T). Part of any MFCC-SVM degradation on
  2021 will therefore be mechanical rather than evidence about replay realism —
  testable by restricting 2019 dev to short clips.
- **Dev has now absorbed heavy selection pressure.** 112 SVM configurations plus 7 RF
  fits were evaluated against it and the minimum reported. Best-of-112 is
  optimistically biased; the top-5 spread (9.216-9.423%) suggests roughly 0.2 pp of
  selection optimism. 2021 eval remains genuinely untouched, so the headline number
  stays clean, but **dev EER must be presented as a tuning number, not a
  generalisation estimate** — and Phase 6 will add further dev evaluations on top.
- **Phase 7 cost is dominated by feature extraction, not scoring.** SVM scoring of all
  721,332 2021-eval files projects to ~0.7h (3.56 ms/file at 37,943 support vectors),
  while extracting features for them costs ~4h at Phase 4's measured rates.
- **Model sizes for the Phase 9 demo**: `svm_mfcc.joblib` is 37MB and scores in
  3.56 ms/file — viable for a live demo. `models/` as a whole is now ~1.6GB, and
  `rf_mfcc_full.joblib` alone is 483MB — a lab artifact, not something to ship.

---

## Phase 6 — CQT-LCNN main system

This is the thesis's actual contribution: the system whose front-end (CQT) is the
variable under test, against Phase 5's MFCC baseline. Four new modules
(`pack_features.py`, `datasets.py`, `models_lcnn.py`, `train_lcnn.py`), seven
training runs, **51.7 hours of GPU time**.

**Headline: dev EER 0.798%, versus the MFCC-SVM's 9.216% -- an 11.6x reduction
(91.3% relative).** ROC-AUC 0.9996 versus 0.9684.

### 6.0 Hardware constraints, measured before designing anything

Three numbers shaped every subsequent decision, and none were what the plan assumed:

- **System RAM is the binding constraint, not VRAM: 5.9 GB total, ~0.7 GB free.**
  The GPU (GTX 1650, 4.29 GB VRAM, ~3.6 GB usable, CUDA 12.1, torch 2.5.1) never
  came close to being the bottleneck.
- **E: is an NVMe SSD** (WDC SN530), so random reads are cheap -- which turned out
  to matter less than expected, see 6.1.
- Compute capability 7.5, but on the **TU117** die -- see 6.5.

### 6.1 The I/O problem, and `src/pack_features.py`

The naive approach -- one `.npy` per file, loaded on demand -- is unusable here, and
the reason is not what it appears to be. Measured cold on the real cache:

| approach | per file | per epoch (175,959 files, 1 worker) |
|---|---|---|
| `np.load` | 10.32 ms | **30.3 min** |
| `np.load(mmap_mode="r")` | 10.41 ms | 30.5 min -- **no benefit** |
| random reads inside ONE open file | **0.27 ms** | **~0.8 min** |
| warm re-read (page cache) | 0.17 ms | -- |

mmap not helping is the diagnostic: if the cost were *data transfer*, memory-mapping
would help. It doesn't, so the cost is **per-file-open overhead** -- syscall, Windows
Defender inspection, and `.npy` header parsing -- paid 175,959 times per epoch.
Reading inside an already-open handle is ~38x faster. And the page cache cannot
rescue this: 4.6 GB of training data against 5.9 GB of total system RAM.

So Phase 6 begins with a packing step. `pack_features.py` concatenates every file's
CQT into **one contiguous `uint8` blob per split**, with a parquet index carrying
`(filename, speaker_id, label, attack_id, env_id, offset, n_frames)`. Reading sample
*i* becomes `seek(offset[i]) -> read(90 * n_frames[i]) -> reshape(90, n_frames[i])`.
Records are variable-length (files genuinely differ), so the index is mandatory --
position cannot be computed arithmetically.

Result: **4.50 GB train + 1.65 GB dev**, built in ~40 min, and measured end-to-end
through the Dataset at **0.334 ms/sample = 0.98 min/epoch, a 31x speedup**.

The packer verifies 200 random files per split **byte-for-byte against the original
`.npy`** before declaring success. A silent offset bug here would corrupt every
training batch with no error surfacing anywhere, so this check is worth its few
seconds. The original per-file cache is deliberately kept (E: has room), so the pack
can be rebuilt without re-running Phase 4.

### 6.2 `src/datasets.py` -- windowing, and why each choice

**Fixed length.** A CNN batch requires identical shapes, but clips differ (median 267
frames, range 92-1059). Every sample is forced to `CQT_FIXED_FRAMES`.

**Short clips are TILED, not zero-padded.** Zero-padding appends a block of digital
silence, which never occurs in real recordings -- it creates a sharp artificial
boundary that a CNN will happily learn to detect instead of learning about replay.
Tiling keeps the input statistically audio-like throughout.

**Long clips are RANDOM-cropped when training, CENTRE-cropped otherwise.** The random
crop is fresh every epoch, which is the entire reason Phase 4 cached variable-length
arrays rather than baking in a fixed window. The deterministic centre crop keeps dev
EER reproducible run to run.

**SpecAugment** (training only) zeroes random frequency bands and time spans, forcing
redundant representations -- essentially dropout in the input's structured domain.
Flagged caveat: frequency masking could mask exactly the high-frequency band the
thesis argues carries the fingerprint, so mask widths are config parameters to tune,
not ASR defaults to copy.

**Normalisation** is `x/255` (preserving everything, including absolute level) or
per-utterance per-bin CMVN. See 6.8 -- this became an experiment rather than a choice.

Each sample is returned as `(1, 90, T)`: the leading 1 is the **channel** axis, since
`Conv2d` expects `(batch, channels, height, width)` and a spectrogram is a
single-channel image.

### 6.3 `src/models_lcnn.py` -- LCNN-9

Layer sizes follow the original LCNN recipe, which is also the backbone of the
official ASVspoof **LFCC-LCNN** baseline. That choice is deliberate and is about
experimental design, not convenience: keeping the backend identical to a published
baseline means the eventual comparison **isolates exactly one variable, the
front-end**. Substituting e.g. ResNet-18 would change two things at once and make
any difference unattributable -- which would gut the thesis's central claim.

**Max-Feature-Map (MFM)** is the defining component. Given `2N` channels it splits
them in half and takes the elementwise maximum, halving the channel count. Where
ReLU compares each activation against a *fixed* threshold of zero (destroying
everything negative), MFM compares two *learned* feature maps against each other and
keeps the winner. Three consequences: it is a competitive, data-driven feature
selection rather than a fixed rule; informative negative-going responses survive;
and there are no dead units, since gradient always flows to whichever branch won.
This suits replay detection specifically, where the evidence is subtle
small-magnitude spectral deviation that a hard zero threshold could discard. The
halving is also what makes LCNN "light" -- each conv sees half the channels the
previous one produced.

Nine conv layers, four 2x2 pools -- note the convs are shape-neutral (padded), so
pooling alone controls resolution while the convs and MFM control channel depth. At
T=400 the frequency axis goes 90->45->22->11->5 and the time axis 400->200->100->50
->25, giving a final `(32, 5, 25)`. A final unit's **receptive field is
64x64 input pixels**: 64 of 90 frequency bins and 64 frames (~1.0 s of audio). The
1x1 convs are channel-mixing bottlenecks -- 9x cheaper than 3x3 -- that reorganise
channels before the expensive spatial filtering.

**Two heads are implemented**, and this is a real experimental axis (6.7):
- `timepool` -- `AdaptiveAvgPool2d((F,1))`, collapsing time to give `C*F = 160`
  features **independent of T**, preserving the frequency axis.
- `flatten` -- the paper's original head, `C*F*T` features, so parameter count
  **scales with T**.

The post-conv shape is **inferred by pushing a dummy tensor through the stack**
rather than hardcoded, so changing T or the bin count cannot silently produce a
wrong `Linear` size -- a failure that would surface as a confusing mid-training
dimension error, or worse, as a wrong-but-runnable layer.

### 6.4 `src/train_lcnn.py` -- protocol

`BCEWithLogitsLoss(pos_weight=3.699)` on a single logit. `pos_weight` is
`n_spoof/n_bonafide`, the same imbalance strategy as Phase 5's
`class_weight="balanced"`. The fused "WithLogits" form is numerically stable: a
separate sigmoid can saturate to exactly 0 or 1 in float precision, making `log(0)`
produce NaN.

**Single logit rather than 2-class softmax**: for binary classification these are
near-equivalent (a 2-class softmax depends only on `z1 - z0`, so it learns one
number parameterised by two). Single logit gives the EER score directly with no
post-processing and removes a place where a sign error could silently invert every
metric.

Adam with `ReduceLROnPlateau` **scheduled on dev EER, not dev loss**. Under 3.7:1
imbalance the loss is dominated by the majority class and can improve while the
bonafide-vs-spoof *ranking* -- which is what EER measures and what we report -- gets
worse. This mattered in practice: in the T=250 run the first LR halving at epoch 24
immediately took dev EER from ~4.4% to 2.8%.

Two checkpoints per run: `_best.pt` (weights at best dev EER, for Phase 7) and a
rolling one carrying **optimizer, scheduler and scaler state** for exact resumption.
The latter made the T400 extension possible (30 -> 45 epochs, continuing from the
existing LR level rather than restarting hot).

**Train EER is measured on a 20,000-file training subsample with augmentation OFF and
a centre crop** -- i.e. scored exactly like dev. Measuring it under augmentation
would compare two different tasks and make the train/dev gap meaningless. That gap
was the planned augmentation diagnostic (see 6.9 for why it could not answer the
question it was built for).

### 6.5 Infrastructure findings (all measured, all contradicted expectations)

**AMP is 2.8x SLOWER on this GPU, not faster.** 227.8 ms/batch with AMP versus
82.3 ms without. The GTX 1650 uses the **TU117** die, which -- unlike the rest of the
Turing family -- **has no tensor cores**, so FP16 buys no arithmetic speedup while
still paying the cast overhead. Its real benefit is memory (peak 0.40 GB vs 0.84 GB),
and memory was never the constraint. `LCNN_USE_AMP = False`. The `GradScaler` is kept
in the loop as a transparent pass-through so the code is identical either way and one
config flag would re-enable it on tensor-core hardware.

**Batch 32 is optimal; 128+ falls off a cliff.**

| batch | ms/sample | peak VRAM | est. min/epoch |
|---|---|---|---|
| 32 | 2.60 | 0.84 GB | **7.6** |
| 64 | 2.97 | 2.32 GB | 8.7 |
| 128 | 21.55 | 4.61 GB | 63.2 |
| 256 | 45.32 | 6.10 GB | 132.9 |

128 and 256 report *more allocated memory than the card physically has* (4.29 GB),
meaning PyTorch spills into shared host memory -- doubly punishing on a machine with
5.9 GB of system RAM. An 8-25x slowdown, not a graceful degradation.

**DataLoader workers fail outright.** `num_workers=2` raised Windows **error 1455
(`ERROR_COMMITMENT_LIMIT`, "the paging file is too small")**. Workers are separate
processes that return batches through shared-memory file mappings, and 5.9 GB total
RAM cannot commit them. `LCNN_NUM_WORKERS = 0`; the cost is ~11%, since loading is
10.7 ms/batch against 83 ms of GPU work.

### 6.6 The T sweep -- context dominates everything

Run with the `timepool` head specifically so parameter count stays **constant across
T** (184,017). With `flatten` it would have grown 388,817 -> 542,417 -> 798,417,
confounding "more context" with "more capacity".

| T | window | dev EER | train EER | best epoch |
|---|---|---|---|---|
| 150 | 2.4 s | 6.584% | 4.522% | 13 (early-stopped 21) |
| 250 | 4.0 s | 2.780% | 1.190% | 28 |
| 400 | 6.4 s | **0.902%** | 0.122% | 43 (extended to 45) |

A **7.3x range** -- larger than every other factor combined.

**T=150 UNDERFITS rather than overfitting**: its train EER (4.522%) is also terrible.
That rules out the competing explanation, because T=150 actually receives *more*
augmentation diversity (with a 267-frame median, nearly every file is randomly
cropped at T=150 versus about half at T=250). It lost anyway, so **context is worth
far more than crop diversity**. Mechanistically this fits the 64-frame receptive
field: T=150 gives ~2 receptive fields of context to integrate, T=400 gives ~6, and a
loudspeaker's frequency signature is stationary enough that evidence accumulates.

**This overturned a recommendation made in this document.** Section 4.3c argued for
T~250 to minimise the 2019/2021 padding mismatch. The measurement says 400,
decisively. The mismatch concern remains real but is a *Phase 7 transfer* risk, not
an in-domain one -- see 6.10.

### 6.7 Head: flatten beats time-pooling

| | timepool | flatten |
|---|---|---|
| dev EER | 0.902% | **0.798%** |
| train EER | 0.122% | 0.238% |
| epochs to best | 43 | **23** |
| total time | 11.08 h | **7.32 h** |
| params | 184,017 | 798,417 |

The prediction recorded before running this was that the 4.3x larger head would
overfit. **It did the opposite**, and the diagnostic explains why: timepool reaches a
*better* train EER while generalising *worse*. Capacity was never the problem --
**pooling destroys information**. Averaging across time discards *where in the
utterance* each feature fired, and that localisation carries real signal, plausibly
transient loudspeaker behaviour at onsets that a time-average smears away.

To make this comparison airtight, timepool was extended past its 30-epoch cap (it was
still improving when truncated, while flatten had genuinely converged with 7 flat
epochs). The extension gained 11.8% relative (1.023% -> 0.902%) across two further LR
drops and then plateaued -- flatten still wins, on final EER, on epochs required, and
on wall-clock.

Choosing `timepool` for the T sweep was nonetheless correct: it kept parameters
constant. `flatten` is only a fair comparison once T is fixed. The two decisions do
not conflict.

### 6.8 CMVN removes the evidence

| | `unit` (x/255) | CMVN |
|---|---|---|
| dev EER | 0.902% | 1.293% |
| train EER | 0.122% | 0.330% |
| final train loss | -- | higher |

CMVN costs **43% relative** at a matched (timepool) head. The decisive detail is that
it is worse **on training data too**, which distinguishes two explanations: a
regulariser fits train worse but generalises better, whereas CMVN fits train worse
*and* dev worse. It is **destroying information, not regularising**.

And the information it destroys is precisely identifiable: the per-utterance,
per-frequency-bin mean -- the stationary spectral level in each band, which is exactly
what a loudspeaker and microphone's frequency response imposes on a recording.

**This is direct mechanistic support for the thesis's central premise**: the replay
fingerprint lives substantially in the stationary per-band spectral level, and
removing it measurably degrades detection. Stronger than citing the claim from
literature, because it is measured on this system. The modest magnitude also matters
for honesty -- at 43% the channel signature clearly contributes but is not the whole
story, since the model still reaches 1.293% without it.

### 6.9 Waveform augmentation -- a clean dose-response, and an unresolvable question

`src/augment_waveform.py` pre-computes augmented copies of the **training split only**
(dev/eval are always evaluated clean). Each copy is an independent random draw: every
file gets its own randomly-sampled chain, with randomised parameters, from:

| effect | stands in for | detail |
|---|---|---|
| additive noise | real mic self-noise | SNR 10-30 dB, half low-pass filtered so it is not always spectrally flat |
| RIR convolution | rooms beyond the shoebox model | exponentially-decaying noise, RT60 0.08-0.5 s, deliberately NOT a shoebox |
| soft clipping | loudspeaker nonlinearity | `tanh` saturation or hard clip; simulation's device responses are perfectly linear |
| MP3/AAC round-trip | recording-chain processing | 64-192 kbps via ffmpeg; codecs discard "inaudible" high-frequency detail -- exactly where the fingerprint lives |

Effect order is shuffled, since *clipping then room* is physically different from
*room then clipping*.

**Why pre-compute rather than augment on the fly**: these need the waveform, but
Phase 4 cached CQTs. Re-decoding every epoch costs ~1 h/epoch (~20 h per run), and
Phase 6 involved seven runs -- so on-the-fly would have cost ~140 h against a one-time
~1.3 h. The performance gap is also smaller than it appears, because fresh random
crops and SpecAugment are applied on top regardless, so the model never sees a
byte-identical input twice under either scheme.

**Augmented waveforms are trimmed back to the original length before the CQT**, so
every copy shares the existing index -- identical offsets, identical frame counts.
`cqt_train_aug1.dat` is byte-identical in size to `cqt_train.dat` (4,500,426,060).
This makes all four blobs interchangeable in layout, and is also what allowed
within-copy resume from file size alone.

At load time the Dataset draws **uniformly at random among {clean, aug1..N}, per
access, independently** -- not a cycle. The same file may draw `aug1, clean, aug3,
aug1` across epochs. Over 30 epochs the chance a file never sees a given variant is
`0.75^30 ~ 0.018%` (~31 of 175,959 files), so coverage is effectively complete without
enforcing it. Keeping **clean in the pool** is deliberate: training exclusively on
perturbed audio would adapt the model to a distribution neither dev nor 2021 has.

Results, all at flatten/T400:

| clean fraction | dev EER | train EER |
|---|---|---|
| 100% (none) | **0.798%** | 0.238% |
| 50% (1 copy) | 1.486% | 0.394% |
| 25% (3 copies) | 2.353% | 0.986% |

**Monotonic: every increment of augmentation costs in-domain accuracy.**

**But this cannot settle whether augmentation succeeded**, and that limitation is
structural rather than a shortcoming of the runs. Augmentation targets the
**simulation shortcut** -- 2019 PA is entirely simulated, so a high-capacity CNN can
score well by keying on simulator regularities (one small family of synthetic RIRs
and device curves) that will not exist in 2021's real re-recordings. **Dev is also
simulated 2019 data.** A model leaning on those regularities therefore looks *better*
on dev, not worse. In-domain degradation is exactly what forcing a model off a
still-profitable shortcut looks like.

So the train/dev gap diagnostic designed in the plan measures only **within-domain
memorisation**, and is blind by construction to the thing waveform augmentation
exists to fix. **Only 2021 can resolve it** -- which is why all three points on the
axis are carried into Phase 7 as pre-registered systems.

### 6.10 Confounds checked

**Padding asymmetry (checked and dismissed).** Bonafide files are systematically
longer than spoof (train 323 vs 274 frames; dev 337 vs 266), so the fraction needing
tile-padding diverges with T:

| | bonafide padded | spoof padded | asymmetry |
|---|---|---|---|
| T=150 | 2.3% | 2.3% | +0.1 pp |
| T=250 | 34.8% | 40.5% | -5.7 pp |
| T=400 | 76.1% | 94.3% | **-18.2 pp** |

Tiling creates exact periodicity, which a CNN can detect -- so "is this repeating?"
could act as a proxy for "is this spoof?". Alarmingly, the asymmetry ordering matches
the performance ordering exactly. **Quantified and ruled out**: duration alone scores
**41.494% EER** (AUC 0.6335) and the tiling factor alone **48.329%** -- essentially
chance, against the model's 1.023%. A model exploiting only that cue could not reach
1%. The T-sweep gain is genuine context benefit. Worth recording in the thesis as a
confound explicitly tested rather than assumed away.

**A latent seed collision (found, assessed, left alone).** Augmentation seeds are
`RANDOM_SEED + 100000*copy + i`, but there are 175,959 files -- so copy 1's tail
overlaps copy 2's head (copy1/file100000 and copy2/file0 both get seed 200042).
Harmless in the way that matters: no single *file* ever gets the same seed in two
copies (those differ by exactly 100,000), so the three copies of any file remain
genuinely different -- verified by hashing. The collision is between *different files
in different copies*, which share augmentation parameters applied to different audio.
~76,000 such pairs; negligible effect on parameter diversity. Not worth a 1.3 h
rebuild that would also invalidate the trained augmented models. Fix if ever
regenerating: multiplier `1_000_000`.

**Augmentation determinism (verified).** Per-file seeding makes `--force` rebuilds
byte-identical, confirmed by hashing, including through ffmpeg codec round-trips. The
per-task seeding is load-bearing: with a single shared RNG the 8 joblib workers would
consume random numbers in nondeterministic interleaving, so results would vary run to
run despite a "fixed" seed.

### 6.11 Full results

| run | T | variant | params | dev EER | train EER | ratio | best ep | ROC-AUC | hours |
|---|---|---|---|---|---|---|---|---|---|
| T150 | 150 | timepool | 184,017 | 6.584% | 4.522% | 0.69 | 13 | 0.9808 | 2.19 |
| baseline_T250 | 250 | timepool | 184,017 | 2.780% | 1.190% | 0.43 | 28 | 0.9959 | 4.79 |
| T400 | 400 | timepool | 184,017 | 0.902% | 0.122% | 0.14 | 43 | 0.9995 | 11.08 |
| cmvn_T400 | 400 | timepool+CMVN | 184,017 | 1.293% | 0.330% | 0.26 | 30 | 0.9989 | 7.73 |
| **flatten_T400** | 400 | **flatten** | 798,417 | **0.798%** | 0.238% | 0.30 | 23 | **0.9996** | 7.32 |
| flatten_T400_aug1 | 400 | flatten+wavaug x1 | 798,417 | 1.486% | 0.394% | 0.26 | 41 | 0.9987 | 11.26 |
| flatten_T400_aug | 400 | flatten+wavaug x3 | 798,417 | 2.353% | 0.986% | 0.42 | 29 | 0.9968 | 7.30 |

Winner `flatten_T400`: dev EER **0.798%**, confusion matrix balanced at the EER
threshold (FNR 0.796% = 112/14,067; FPR 0.799% = 408/51,030).

**Comparison against Phase 5** (same dev split, same metric, same convention):

| system | dev EER | ROC-AUC |
|---|---|---|
| MFCC-SVM (RBF, tuned, full train) | 9.216% | 0.9684 |
| MFCC-RF (full train) | 11.736% | 0.9548 |
| **CQT-LCNN (flatten, T=400)** | **0.798%** | **0.9996** |

### 6.12 Caveats to carry into the write-up

- **Dev has absorbed heavy selection pressure.** Phase 5 evaluated 112 SVM
  configurations against it; Phase 6 adds seven more systems plus per-epoch
  checkpoint selection. Dev EER is a *tuning* number, not a generalisation estimate.
  2021 remains untouched and is the only clean figure.
- **Unequal epoch budgets.** `aug1` ran 45 epochs against its control's 30 (each to
  convergence, but budgets differ), and `aug3` was truncated at its cap while still
  creeping down -- extended it might reach ~2.2%. Neither changes the monotonic
  conclusion, but both should be stated.
- **T=400 maximises the 2019->2021 mismatch.** 2021 clips average ~149 frames, so at
  T=400 essentially every 2021 file is tiled ~2.7x versus ~1.4x on 2019. The
  best in-domain configuration carries the largest transfer risk, and that cannot be
  measured without touching 2021. Hence the pre-registration in
  `PROJECT_PLAN.md` phase 7: T=400 is the primary system, T=250 a documented
  robustness check, with the prediction *stated in advance* that T=250 may transfer
  better. Registering it beforehand means either outcome is an honest result rather
  than eval-tuning.

### 6.13 Artifacts

Results are now grouped per phase, and within Phase 5 per model family, rather than
distinguished only by filename prefix:

```
results/phase5/summary.{md,json}
             /svm/  sweep.csv, sweep.png, grid_search_legacy.csv, dev_scores.csv
             /rf/   curve.csv, feature_importance.png, n_estimators.{csv,png}, dev_scores_*.csv
results/phase6/<tag>/  log.csv, curves.png, summary.json, dev_scores.csv
models/phase5/  svm_mfcc.joblib, rf_mfcc_full.joblib, rf_mfcc_sub_<n>.joblib
models/phase6/  lcnn_<tag>.pt, lcnn_<tag>_best.pt
E:/ASVspoof/packed/  cqt_{train,dev}.dat + indices, cqt_train_aug{1,2,3}.dat
```

`models/` (1.7 GB) and every `dev_scores*.csv` are gitignored. Note the ignore
pattern needed `**/dev_scores*.csv`: a single `*` does not cross directory
separators, so the previous `results/*dev_scores*.csv` silently failed to match the
nested Phase 6 files.

---

## Phase 7 — Evaluation on held-out 2021 PA (COMPLETE)

**The held-out set was scored once, as pre-registered.** 943,110 files, nine of our
systems plus the four official ASVspoof baselines, zero extraction failures.

**Headline: `flatten_T400_aug` reaches 32.665% EER on 2021 PA eval, beating all four
official baselines (best of which is CQCC-GMM at 38.068%) by 5.40 pp / 14.2%
relative. Five of our seven CQT-LCNNs beat every official baseline.** Every system
also degraded enormously from dev — the pre-registered primary `flatten_T400` went
from 0.798% to 39.747%, a 50x collapse. Both facts are the result: the 2021 PA task
defeats the entire published baseline set, and our systems are defeated less.

Sections 7.0–7.9 record the build, the engineering, and three separate machine
limits hit along the way. **Results begin at 7.10.**

### 7.0 A gap in the pre-registration, found before the run

The registration in `PROJECT_PLAN.md` listed five LCNNs, and **could not test two of
its own three predictions**. `cmvn_T400` and `baseline_T250` are both *timepool*
models while the primary `flatten_T400` is *flatten*, so comparing either against the
primary varies **two** things at once — normalisation *and* head for prediction 3
("CMVN may transfer better"), T *and* head for prediction 1 ("T=250 may transfer
better"). Either result would have been uninterpretable.

The fix costs almost nothing: **`T400` (timepool, unit) was already trained in Phase 6**
(0.902% dev EER) and is the matched control — `cmvn_T400` differs from it only in
normalisation, `baseline_T250` only in T. Added along with `T150` (completing the T
axis at 150/250/400, all timepool, ~5 min of GPU) and the Phase 5 **MFCC-RF**
(~6 min from cached MFCC). Nine systems total, all frozen in Phase 5/6, all declared
in `src/config.py` **before any 2021 score existed**. The primary remains
`flatten_T400` regardless of outcome.

The general lesson, worth keeping: a pre-registered comparison is only as good as its
*control*. Registering the interesting systems is not enough if nothing in the list
isolates one variable at a time.

### 7.1 Modules

- **`src/evaluate_2021.py`** — the one expensive pass. Streams the corpus in chunks
  of 4,000: 8 joblib workers decode → MFCC → CQT, then the main process windows each
  chunk and scores all 7 LCNNs. The seven need only **four** distinct inputs
  ((T400,unit), (T400,cmvn), (T250,unit), (T150,unit)), so each CQT is windowed four
  times rather than seven, with four models sharing the (T400,unit) view. Resumable
  at chunk granularity; caches CQT and pooled MFCC; verifies the cache byte-for-byte.
- **`src/score_classical_2021.py`** — SVM and RF from the cached MFCC. A *separate*
  pass on purpose: libsvm scoring is single-threaded CPU work that would otherwise
  contend with the 8 extraction workers for the same cores, slowing the expensive
  pass to speed up the cheap one.
- **`src/report_2021.py`** — reads score tables only, never audio, so every table and
  figure regenerates in seconds. Covers 7.1–7.5 plus both controls and an explicit
  verdict on each registered prediction.

### 7.2 The refactor, and how it was proven safe

`_fit_length` and `_normalise` moved out of `CQTDataset` to module level in
`datasets.py`. Phase 7 cannot reuse the class (2021 CQTs are never written into the
Phase 6 blob layout), but if the 2021 windowing diverged from how dev was windowed,
the dev-vs-eval comparison this entire phase exists to make would be invalid. Sharing
one code path makes that a fact rather than an assertion.

Because the change touches training, it was verified rather than eyeballed, in two
stages:

1. **Behaviour fingerprint, before and after.** Eight configurations (dev/train,
   centre/random crop, unit/CMVN, T=150/250/400, with and without the 3-copy waveform
   augmentation) were hashed over deliberately chosen indices spanning both the
   tile-pad and crop branches. **All eight identical.** The `train_random_*` cases are
   the load-bearing ones: `fit_length` and `_specaugment` draw from the same RNG, so
   had the refactor changed how many draws the crop consumes, every subsequent
   SpecAugment mask would have shifted and the hash would have moved.
2. **Phase 6 end to end.** `flatten_T400`, `cmvn_T400` and `baseline_T250` were
   re-scored on dev through the refactored path and reproduce their recorded EERs to
   four decimals — 0.7979 / 1.2926 / 2.7801%, **delta 0.000000 pp** — with per-file
   score agreement to ~1.5e-6. Ten training batches also run clean. The residual 1e-6
   is float32 cuDNN algorithm-selection nondeterminism; EER is rank-based and so is
   unaffected by it.

### 7.3 Validating the Phase 7 scoring path without touching 2021

`evaluate_2021` does not use `CQTDataset` — it windows raw variable-length arrays
itself and batches them through several models at once. That path needed proving on
data where the answer is already known. So dev CQTs were read from the Phase 6 packed
blob **at natural length** (exactly the shape extraction yields), pushed through
`score_chunk` exactly as the 2021 run will, and compared against each model's
`dev_scores.csv`.

**All 7 systems: correlation 1.0000000000, worst deviation 2.7e-5.** The 512-file
sample spanned 119–1089 frames, so both the tile-pad and crop branches were
exercised. Zero 2021 labels involved.

### 7.4 A bug that silently discarded 11% of the data

The first smoke run reported success while dropping **69 of 600 files (11.5%)** —
68 of them `WinError 1455` (`ERROR_COMMITMENT_LIMIT`), the *same* commit-limit
ceiling that rules out DataLoader workers (6.5).

Cause: the extraction worker was defined in `evaluate_2021.py`, which imports torch.
**joblib's loky workers import the module a function is defined in** in order to
unpickle it, so all 8 workers were each loading torch and the CUDA runtime (~300 MB
apiece) — something Phase 4 never did, because its worker lived in the torch-free
`features.py`. Moving the worker to `features.extract_for_eval` fixed it outright:
**0 failures** on re-run.

Two things were changed as a result, and the second matters more than the first:

- The worker's home is now load-bearing, and is documented as such in its docstring —
  it is not a tidiness choice, and moving it back would silently reintroduce the bug.
- **A chunk now counts as complete only when it holds a score for every file in it**,
  not merely when its shard exists, plus a failure-rate guard that aborts loudly
  rather than grinding on. The original design would have absorbed an 11% hole
  straight into a headline EER with nothing surfacing as an error anywhere. Phase 4
  decoded 241,056 files through this same ffmpeg path with zero failures, so any
  sustained failure rate here means something environmental that re-running will not
  fix and that must not be quietly tolerated. `--accept-failures` is the explicit
  escape hatch once a failure is understood.

### 7.4b The interleaved design failed on memory, and was split in two

The first real run died at chunk 2: **89/8,000 files (1.1%)**, tripping the guard from
7.4. All of them were memory exhaustion — 83 × `WinError 1455`, plus ffmpeg exiting
with `0xC000012D` (`STATUS_COMMITMENT_LIMIT`) and `0xC0000142`
(`STATUS_DLL_INIT_FAILED`) — and throughput had fallen to 28 files/s as the machine
paged.

The interleaved design asks a 5.9 GB machine for a parent holding **torch + a CUDA
context + 7 models (~2 GB)** at the same moment as **8 worker interpreters holding
librosa/numba (~250 MB each)**. Phase 4 ran 8 workers safely for exactly the reason
this could not: its parent had no torch in it at all.

Rationing the workers did fix the failures — `--n-jobs 6 --no-prefetch` gave 0
failures — but at **18 files/s**, a 14.6 h run. So the contention was removed instead
of rationed, by splitting into two stages that never coexist:

- **`--stage extract`** imports no torch whatsoever (verified: `'torch' in sys.modules`
  is `False` after importing the module — the torch, `datasets` and `models_lcnn`
  imports are all function-local, since `datasets` pulls torch in transitively). Light
  parent, 8 workers, i.e. Phase 4's proven configuration.
- **`--stage score`** runs no extraction workers and reads CQT back from the cache.

The corpus is still decoded exactly once. This is only possible *because* the CQT is
cached — a decision justified independently as insurance in 7.6, which has now paid
for itself twice.

### 7.4c ffmpeg was the bottleneck, and Phase 4's decode rule was over-general

With the memory problem solved, extraction still ran at only **20 files/s** — a mere
**1.1x** over single-threaded, when 6 physical cores should give ~6x. Cores were idle,
so the work was latency-bound, not compute-bound. Measured on 300 cold files each:

| | 1 worker | 8 workers | speedup |
|---|---|---|---|
| raw disk read (no decode) | 10.5 ms | **2.0 ms** | 5.3x — scales fine |
| **ffmpeg bare spawn** (`-version`, no input file) | **20.2 ms** | — | — |
| full decode | 41.7 ms | **20.9 ms** | 2.0x — stalls |

Decode at 8 workers costs 20.9 ms/file and launching ffmpeg *with no input at all*
costs 20.2 ms. **Decode was essentially 100% process-spawn overhead**, and spawn is a
system-wide serialisation point (kernel plus Defender scanning the binary on every
launch), so it barely parallelises. That capped decode near ~48 files/s regardless of
core count — which is why 8 → 12 → 16 workers never helped, and why disk was never
the issue at ~1 MB/s effective.

**The fix came from re-testing a Phase 4 assumption.** Phase 4 measured soundfile
failing on ~46% of the corpus and switched everything to ffmpeg for one uniform decode
path. That 46% was measured on **2019** and never re-tested on 2021 — where soundfile
reads **500/500** sampled files. `load_audio` now tries soundfile first and falls back
to ffmpeg for anything it cannot read, so correctness never depends on soundfile's
coverage; only speed does.

**This is not a change in numerical behaviour, and that was verified rather than
argued.** FLAC is lossless, so any correct decoder must agree — and it does:

- 2021, both decoders succeed: **120/120 bit-identical, max abs diff exactly 0.0**
- 2019 sample of 150 (soundfile succeeds on 51%, the rest exercising the fallback):
  **150/150 bit-identical**, and the recomputed CQT matches **Phase 4's existing
  on-disk cache byte-for-byte, 150/150**

So Phase 4's cache remains exactly reproducible through the new function, and the
"one uniform decode path" rule is *satisfied in the sense that mattered* — it existed
to prevent numerical inconsistency between decoders, and there is provably none. The
honest restatement for the thesis is that the two decoders are interchangeable on this
data, verified, and the choice is now made on speed alone.

Result: extraction **20 → 64.5 files/s, a 3.2x speedup**, taking it from 12.3 h to
~4.1 h.

The general lesson, which is the same one Phase 5's dropped-features error taught:
a measurement made once, on one corpus, under one set of conditions, silently becomes
an assumption everywhere else. The 46% figure was correct and load-bearing when it was
made; it was simply never re-checked against the data it was later applied to.

### 7.5 Costs, measured — including one estimate that was wrong twice

- **A benchmarking artifact, caught.** A first pass measured 175.5 ms/file and put
  MFCC at 104.5 ms — which would have made 2021's *shorter* clips cost more per file
  than 2019's longer ones, an obvious contradiction. It was librosa's lazy filterbank
  construction on first call. Warm: **decode 28.7 ms + MFCC 3.6 ms + CQT 23.2 ms =
  55.5 ms/file**.
- **An early 44 files/s reading was page-cache-warm** and should not have been
  trusted: it came from repeatedly re-running `--limit 12000` over the *same* first
  12,000 files, so the OS had them cached. The honest cold-disk figure for the
  interleaved design was ~20-28 files/s. Worth recording as a benchmarking trap —
  re-running a benchmark over an identical file set measures the page cache, not the
  workload.
- Pinning BLAS/FFT to one thread per worker gained 11% (36 → 40 files/s, applied
  automatically), and 12 workers measured no faster than 8. Neither mattered next to
  the decode fix in 7.4c.

**Final measured throughput, after the two-stage split and the decoder change:**

| stage | rate | full 943,110 |
|---|---|---|
| extract (8 workers, no torch in process) | 64.5 files/s | ~4.1 h |
| score (GPU, reading the CQT cache) | 133 files/s | ~1.9 h |
| classical SVM + RF (from cached MFCC) | — | ~1.0 h |

**Total ~7 h**, against ~4.7 h originally sketched — but arrived at by measurement,
with the two designs that would have taken 13.8 h and 14.6 h eliminated on evidence.

### 7.6 The 27GB caching decision was based on a wrong number

`PROJECT_PLAN.md` 4.4 rejected caching 2021 CQT at an estimated ~27 GB. That assumed
96 bins × 400 *padded* frames. Neither holds — `n_bins` is 90 (the Nyquist fix), and
features are cached at natural length. At the measured mean of 149.4 frames the real
cost is `90 × 149.4 × 943,110 = **12.7 GB**` for all partitions, against 32 GB free.
Less than half the number the decision rested on.

It is now cached, justified as **insurance rather than speed**: the dominant risk is
finding a bug *after* a 6-hour pass, and cached, re-scoring all nine systems costs
~1.5 h of GPU and no CPU. Written as one blob shard per chunk — a monolithic blob
would need a ~25 GB merge and a transient 25.4 GB footprint against 32 GB free — with
a `shard` column in the index. Pooled MFCC is cached too (~500 MB, 0.05% of the CQT
figure). Cache integrity is verified by re-extracting random files and comparing
byte-for-byte, as `pack_features.py` does: **25/25 identical** in the smoke run.

### 7.7 A result already in hand, from 2019 dev only

The short-clip control (`results/phase7/control_short_clips_2019dev.csv`) needs no
2021 data, so it is already valid. Restricting 2019 dev to 2021-like durations:

| system | all dev | ≤250 frames | ≤200 frames | ≤150 frames |
|---|---|---|---|---|
| MFCC-SVM | 9.216% | 13.139% | 15.120% | **17.021%** |
| CQT-LCNN `flatten_T400` | 0.798% | 0.975% | 1.148% | **1.709%** |
| n files | 65,097 | 27,813 | 11,878 | 1,999 |

So **a 1.85x MFCC-SVM degradation is available from clip length alone**, with no
domain shift whatsoever — which is exactly the mechanical 1/sqrt(T) pooling-noise
effect predicted at the end of Phase 5. Any 2021 degradation must have this
subtracted before it is attributed to simulated-vs-real replay.

The LCNN result is the more interesting one and was *not* predicted: it degrades
**2.14x**, proportionally more than the SVM. The Phase 5 prediction was specifically
about pooled *statistics* getting noisier, which should not apply to a CNN consuming a
spectrogram. The mechanism is different — a shorter clip supplies fewer genuinely
distinct frames before tiling repeats them, so at T=400 a 150-frame file delivers
~2.7 copies of the same evidence rather than 2.7x more evidence. Context, which 6.6
showed to be worth more than every other factor combined, is what is actually being
lost. This sharpens the transfer risk already flagged in 6.12 and makes prediction 1
a genuinely open question rather than a formality.

### 7.8 Two conventions for the condition breakdown, because the metadata forces it

Verified directly against `trial_metadata.txt`: `room` (R1–R9) and `mic` (M1–M3) carry
both classes, but `r`/`m`/`s`/`c` are `-` on **every** bonafide row — they describe the
replay device, and a bonafide recording was never replayed through anything. `dist`
splits by class too (`D1–D6` bonafide, `d1–d6` spoof), so the two are different
physical quantities and must not share an axis.

A spoof-only group has no FRR curve and therefore **no EER at all**, so some pooling is
forced. `room`/`mic` get an ordinary within-group EER; replay-device factors pool *all*
bonafide against each condition's spoof, which makes FRR(θ) identical in every group
and lets only FAR(θ) move — so a difference between conditions is attributable purely
to how detectable that condition's attacks are, and cannot be an artefact of one group
holding easier genuine speech. **Caveat to carry into the write-up**: the shared
bonafide half makes those EERs statistically *correlated*, so no test assuming
independent groups applies. FAR at the single global EER threshold is reported
alongside, since it needs no convention explained.

### 7.9 Artifacts

```
results/phase7/   eer_table_2021.csv, eer_other_partitions.csv,
                  condition_breakdown.{csv,png}, det_curves_2021.png,
                  eer_comparison_2021.png, control_short_clips_2019dev.csv,
                  control_duration_cue_2021.csv, registered_predictions.csv,
                  summary.json
                  scores/<system>.score.txt   (official ASVspoof format; gitignored)
E:/ASVspoof/phase7_2021/   cqt/ (blob+index shards), mfcc/, lcnn_scores/,
                           classical_scores/, merged parquets
```

`score.txt` exports are one per system in the official `FILENAME SCORE` format, same
orientation as the four official baselines, so an examiner can recompute every
reported number from a ~20 MB text file with no corpus, no GPU and no part of this
pipeline.

---

### 7.10 The run itself

Completed clean, at the throughputs measured in 7.5:

| check | result |
|---|---|
| shards (index / blob / mfcc / scores) | 236 / 236 / 236 / 236 |
| files scored | **943,110 / 943,110 (100.000%)** |
| extraction failures | **0** |
| missing vs manifest / duplicate filenames | 0 / 0 |
| CQT cache vs fresh extraction | **300/300 byte-identical** |
| stored scores vs fresh re-extract + re-score | **PASS**, worst 2.7e-5 |
| NaN / inf across all 9 systems | 0 / 0 |
| classical scores, fresh re-score of 200 files | **PASS**, worst 5.0e-8 |
| disk | 12 GB CQT + 616 MB MFCC + 36 MB scores = **12.7 GB** (predicted 12.7) |

Zero failures across 943,110 files validates the decode path at ~4x the scale Phase 4
established it at. The end-to-end check is the one that matters most: it re-decoded 60
files from source, re-extracted their CQT, re-ran all seven models and compared against
what was stored — agreement to 2.7e-5 (float32 cuDNN nondeterminism) means the cache,
the offsets, the windowing and the model loading are sound *as a chain*, not merely
individually.

One cosmetic issue: sklearn stores `verbose` in the pickle at fit time and re-fires it
at predict time, so the Random Forest emitted a joblib banner per shard and buried the
progress bar in thousands of lines. Scores unaffected; `_quieten()` now clears it.

### 7.11 Results — 2021 PA eval (`partition=="eval"`, 721,332 trials)

94,068 bonafide / 627,264 spoof. Every number below is produced by `metrics.py`, the
same module Phases 5 and 6 used, over the same trials, for our systems and the official
baselines alike.

| system | 2021 EER | dev EER | ROC-AUC |
|---|---|---|---|
| **`flatten_T400_aug`** (3 aug copies) | **32.665%** | 2.353% | 0.7398 |
| `flatten_T400_aug1` (1 aug copy) | 34.006% | 1.486% | 0.7221 |
| `T150` | 34.420% | 6.584% | 0.7078 |
| `baseline_T250` | 35.816% | 2.780% | 0.6918 |
| `T400` | 38.031% | 0.902% | 0.6642 |
| CQCC-GMM *(official)* | 38.068% | — | 0.6684 |
| LFCC-GMM *(official)* | 39.540% | — | 0.6475 |
| `flatten_T400` **(pre-registered primary)** | 39.747% | 0.798% | 0.6383 |
| `cmvn_T400` | 43.468% | 1.293% | 0.5855 |
| LFCC-LCNN *(official)* | 44.768% | — | 0.5735 |
| MFCC-RF | 45.833% | 11.736% | 0.5586 |
| RawNet2 *(official)* | 48.605% | — | 0.5188 |
| MFCC-SVM | 49.635% | 9.216% | 0.5066 |

**Five of seven CQT-LCNNs beat all four official baselines.** Best-vs-best is 5.40 pp
(14.2% relative) over CQCC-GMM.

MFCC-SVM at ROC-AUC 0.5066 is statistically indistinguishable from a coin flip: the
classical baseline retains **no usable signal** on real replay. Note also that the
classical ordering *inverted* — RF (45.833%) now beats SVM (49.635%), reversing dev.

Supplementary metrics (precision/recall/F1 at the EER threshold) are in
`results/phase7/eer_table_2021.csv` but are dominated by the 6.67:1 imbalance —
precision_bonafide is ~0.24 even for the best system — so **EER is the number to
report and the rest is context**.

### 7.12 The measurement chain is VERIFIED against the published record

Checked against Liu et al., *"ASVspoof 2021: Towards Spoofed and Deepfake Speech
Detection in the Wild"*, IEEE/ACM TASLP (doi: 10.1109/TASLP.2023.3285283) —
**Table XV**, the full PA progress/evaluation results.

The four official baselines were scored by *our* code, from *their* published score
files, on the trials *we* selected:

| baseline | ours (eval) | paper | ours (progress) | paper |
|---|---|---|---|---|
| B01 CQCC-GMM | 38.068 | **38.07** | 36.331 | **36.33** |
| B02 LFCC-GMM | 39.540 | **39.54** | 39.788 | **39.79** |
| B03 LFCC-LCNN | 44.768 | **44.77** | 42.163 | **42.16** |
| B04 RawNet2 | 48.605 | **48.60** | 46.026 | **46.03** |

**All eight agree to ≤0.005 pp** — pure rounding to the paper's two decimals.

A third, independent confirmation fell out of the hidden-track analysis (7.19),
on differently-defined subsets and against a different table (**Table X**):

| subset | ours | paper Table X (PA) |
|---|---|---|
| eval restricted to D4/d4, "with non-speech" | 30.02 / 32.16 / 49.02 / 43.95 | **30.02 / 32.16 / 49.02 / 43.95** |
| `trim` subset, "w/o any non-speech" | 36.65 / 39.09 / 51.65 / 44.26 | **36.65 / 39.09 / 51.65 / 44.26** |

(B01 / B02 / B03 / B04 in each cell.) Exact on all eight.

Three independent checks, across three differently-constructed trial subsets,
jointly confirm the label join, the `partition` filter, the `trim_flag` and `dist`
semantics, the score orientation, and the EER implementation. **The results chapter
can therefore state that this pipeline reproduces the published ASVspoof 2021 PA
baselines exactly** — which anchors every other number in the thesis, including the
ones no one else has computed.

### 7.13 The registered predictions: 3 of 4 supported

Written down in `PROJECT_PLAN.md` before 2021 was touched, each against a control
differing in exactly one variable.

| prediction | system vs matched control | outcome |
|---|---|---|
| **1.** shorter T transfers better | `baseline_T250` 35.816 vs `T400` 38.031 | **supported, −2.22 pp** |
| **2a.** mild augmentation helps | `aug1` 34.006 vs `flatten_T400` 39.747 | **supported, −5.74 pp** |
| **2b.** aggressive augmentation helps | `aug3` 32.665 vs `flatten_T400` 39.747 | **supported, −7.08 pp** |
| **3.** CMVN transfers better | `cmvn_T400` 43.468 vs `T400` 38.031 | **refuted, +5.44 pp worse** |

**Prediction 2 is the most valuable finding in the project, and it inverted exactly.**
On dev, augmentation was monotonically harmful (0.798 → 1.486 → 2.353). On 2021 the
ordering flips precisely (39.747 → 34.006 → 32.665). Section 6.9 argued this was
*structurally* untestable in-domain, because dev is also simulated 2019 data and a
model exploiting simulator regularities therefore looks **better** on dev. That
argument is now confirmed by measurement rather than asserted. **Optimising on dev
would have discarded the single most effective technique in the project.**

**Prediction 1 holds, and the trend continues past where it was registered.** On 2021:
T150 34.420 < T250 35.816 < T400 38.031 — a clean monotonic inversion of the dev
ordering (6.584 / 2.780 / 0.902). Adding `T400` (timepool) to the registration is what
made this testable; against `flatten_T400` the comparison would have confounded T with
head. That amendment (7.0) earned its place.

**Prediction 3 is cleanly refuted.** CMVN hurt in-domain *and* out-of-domain, so 6.8's
conclusion — that it destroys the per-band spectral level carrying the replay
fingerprint — holds on real replay too. A negative result, and a clean one.

**The dev→2021 rank inversion is suggestive, not established**: Spearman ρ = **−0.607,
p = 0.148** across the 7 LCNNs. Striking and mechanistically explicable, but **n=7 and
it is not significant** — it must be reported as a described pattern with its p-value,
never as a proven law.

### 7.14 The thesis's central claim, on real physical replay

| comparison | margin |
|---|---|
| CQT-LCNN 32.665% vs **MFCC-SVM 49.635%** | 16.97 pp / 34% relative |
| CQT-LCNN 32.665% vs **LFCC-LCNN (official) 44.768%** | 12.10 pp / 27% relative |
| non-augmented `flatten_T400` 39.747% vs LFCC-LCNN | 5.02 pp / 11% relative |

The second row is the closest test the challenge affords to the axis 6.3 chose the
LCNN backbone to isolate. The third row matters because it shows the front-end
advantage does not depend on the augmentation win.

**CORRECTION forced by the paper — B03 is not a plain LCNN.** Table V states B03 is
**LFCC + LCNN-LSTM**. The published score directory is named `LFCC-LCNN`, which is
what misled an earlier draft of this section into claiming the comparison shares our
backbone exactly. It does not: their baseline adds a recurrent stage ours has no
equivalent of. The honest statement is **"same LCNN family, but their baseline adds
an LSTM, so the comparison is front-end-*dominant*, not front-end-*only*"**. The
12.10 pp margin is unaffected; only its interpretation narrows.

**DISCLOSURE that must appear wherever our numbers sit beside challenge numbers.**
The paper (§II, p.2) states participants were *required* to train PA systems on the
**ASVspoof 2019 PA training partition** alone — 54,000 files — and for LA it says
use of the 2019 *evaluation* subset was "strictly forbidden". Our models were trained
on the Phase 2 enriched speaker-disjoint resplit: **175,959 files (~3.3x), and that
pool deliberately includes the entire 2019 PA eval partition** (see section 4 of
`PROJECT_PLAN.md`).

**Our systems are therefore NOT challenge-compliant.** This does not invalidate
anything — this is a thesis, not a challenge entry, and 2019 eval was never our test
set (2021 is, and it stayed untouched until Phase 7). But any table placing our
systems next to challenge systems must state that we trained on ~3.3x the permitted
data, including a partition the rules excluded. An examiner familiar with ASVspoof
will look for exactly this, and it is far better volunteered than extracted.

### 7.15 Condition breakdown — decoded, and it reproduces the published findings

The paper's **Table III** decodes every factor code, and **Table IX** (supplementary)
names the actual hardware. Our columns map as:

| our column | meaning | levels |
|---|---|---|
| `room` | `S_asv`, room for **voice presentation** (ASV side) | R1–R9, sizes in Table III |
| `r` | `S_a`, room for **replay acquisition** (attacker side) | r1–r9 |
| `mic` | `Q_asv,m`, **ASV** microphone | M1 Medium, M2 High, **M3 Low** |
| `m` | `Q_a,m`, **attacker** microphone | m1 Medium, m2 High, **m3 Low** |
| `s` | `Q_a,s`, attacker **replay device** | s2 Low, s3 Medium, **s4 High** |
| `c` | `D_a`, attacker-to-talker distance | c2 1.5 m, c3 1.0 m, **c4 0.5 m** |
| `dist` | `D_s` talker-to-ASV (bonafide) / `D'_s` attacker-to-ASV (spoof) | D1–D6 / d1–d6, d1 2.0 m → d4 0.5 m |

Hardware (Table IX): **s2** = Sony SRS-XB43 (consumer Bluetooth), **s3** = Neumann
KH 80 DSP, **s4** = Genelec 8030 CP (both studio monitors, ±2 dB passband specified;
s2's is listed UNAVAILABLE). **m1/M1** = Marantz MPM-1000, **m2/M2** = M-Audio
Uber-mic (SNR 110 dB), **m3/M3** = iPad Air MEMS mic.

Our results (`results/phase7/condition_breakdown.csv`), against the paper's §III-B-2:

| factor | our result | paper's published finding | |
|---|---|---|---|
| `s` replay device | **s4 (High) hardest, 47.62%** | "min t-DCF is also higher for a better quality attacker replay device `Q_a,s` = s3 or s4" | ✅ |
| `m` attacker mic | **m3 (Low) easiest, 35.36%** | "higher-quality m1 and m2 lead to higher min t-DCFs than m3 … a higher-quality microphone introduces less distortion" | ✅ |
| `c` attacker-to-talker | **c4 (0.5 m) hardest, 42.07%** | "shorter `D_a` also lead to worse performance. At the closest position `D_a` = c4, min t-DCF values are the highest" | ✅ |
| `mic` ASV mic | **M3 (Low) hardest, 42.27%** | conclusions: added difficulty "when the automatic speaker verification microphone is of lower quality" | ✅ |
| `room` size | 10.54 pp spread, no size relation (R7 smallest at 40.06%, R1 largest at 38.16%) | "no substantial correlation between the room size and the min t-DCF values" | ✅ |
| `dist` attacker-to-ASV | closer = **easier** (d4 0.5 m → 37.49%; d1 2.0 m → 41.37%) | "`D'_s` is observed to be inversely correlated with the min t-DCF" — i.e. closer = harder | ❌ **conflicts** |

**Five of six reproduce published findings, using a different metric (EER vs min
t-DCF) and an entirely different system.** That is strong independent corroboration
and belongs in the results chapter as such.

**The mechanism, stated once because it is the thesis's premise made visible twice.**
The same device quality acts in *opposite directions* depending on which side of the
recording chain it occupies. A **low-quality attacker microphone (m3) makes detection
EASIER** — it stamps extra distortion onto the spoof. A **low-quality ASV microphone
(M3) makes detection HARDER** — it degrades the evidence at capture, for bonafide and
spoof alike. Likewise s4 (Genelec, flattest specified passband) is hardest precisely
because a better loudspeaker imposes *less* fingerprint. Note that the top-end
differences between devices (20/22/25 kHz) are irrelevant here — everything above the
8 kHz Nyquist is gone — so what is being detected is passband flatness and
nonlinearity inside 0–8 kHz, exactly the electromechanical signature the thesis argues
CQT preserves.

**The one conflict, reported rather than buried.** For attacker-to-ASV distance our
EER falls as the attacker gets closer, while the paper's min t-DCF rises. Two
candidate explanations, neither verified: (i) the metrics differ, and t-DCF folds in
ASV behaviour that EER does not; (ii) our pooled-bonafide convention (7.8) pools
bonafide spanning *all six* talker positions against spoof at *one* attacker position,
which could leak a level cue that a matched-distance comparison would not. Worth one
paragraph and a flagged uncertainty; not worth over-claiming either way.

**Two findings the paper does not report.** It excluded rooms from analysis after
finding no *size* correlation — but room *identity* matters enormously. The attacker
room `r` has a **24.50 pp spread** (r3 52.45%, worse than chance, vs r5 27.95%), and
grouping by the room-triples the paper defines in its footnote 7
(`{r1,r2,r3} {r4,r5,r6} {r7,r8,r9}`) gives group means of **46.7 / 31.8 / 39.7%**.
Something about the second room group makes replay detection far easier, and it is
not size. This is a genuinely novel observation and a good thesis discussion point.

### 7.16 Both controls came back clean

**Duration is dead as a cue on 2021**: duration alone scores **50.79% EER**, tiling
factor **49.06%** — exactly chance. The tile-periodicity confound quantified and
dismissed on 2019 in 6.10 does not transfer at all, so no model can be exploiting it
here. Note this is *worse than chance-adjacent* 2019 behaviour (41.5%), and recall the
class-duration relationship **inverts** between corpora — 2019 bonafide are longer,
2021 spoof are marginally longer (2.412s vs 2.366s) — so any residual 2019 reliance on
"longer ⇒ bonafide" is actively harmful out of domain.

**The mechanical component is real but small.** Restricting 2019 dev to 2021-like clip
lengths (`results/phase7/control_short_clips_2019dev.csv`):

| system | all dev | ≤250f | ≤200f | ≤150f |
|---|---|---|---|---|
| MFCC-SVM | 9.216% | 13.139% | 15.120% | **17.020%** |
| `flatten_T400` | 0.798% | 0.974% | 1.148% | **1.709%** |

Of MFCC-SVM's 40.4 pp total collapse, only ~7.8 pp is attributable to clip length. So
**roughly 80% of the degradation is genuine simulated→real domain shift**, not
measurement artifact. That is the number the discussion chapter needs, and it is
available *because* the control was built before the results were known.

### 7.17 The `hidden` partition splits the systems in two

Extracted in the same pass as a free consistency check, never used for selection
(`results/phase7/eer_other_partitions.csv`):

| system | progress | eval | hidden |
|---|---|---|---|
| `flatten_T400_aug` | 29.795% | 32.665% | **30.934%** |
| `flatten_T400_aug1` | 32.041% | 34.006% | **30.028%** |
| `T150` | 31.973% | 34.420% | 48.390% |
| `baseline_T250` | 32.683% | 35.816% | 50.078% |
| `T400` | 36.364% | 38.031% | 51.740% |
| `flatten_T400` | 37.537% | 39.747% | 50.795% |
| `cmvn_T400` | 42.525% | 43.468% | 53.882% |
| MFCC-SVM | 49.711% | 49.635% | 52.053% |

`progress` tracks `eval` closely for every system — a clean replication. But on
**`hidden`, every non-augmented system collapses to chance (48–54%) while the two
waveform-augmented systems hold at ~30%.** Section 7.19 decomposes why.

### 7.19 Decomposing `hidden` — the project's most important result

The paper (§IV-A, §IV-B) reveals that PA `hidden` is **two different hidden tracks**:
*hidden track 1* is **simulated** replay data, and *hidden track 2* is data with
**non-speech removed** by VAD. Both are restricted to the D4 talker-to-ASV and d4
attacker-to-ASV positions.

Our metadata separates them exactly. `hidden` is 134,730 rows = **67,365 `notrim` +
67,365 `trim`**, and both halves carry only `dist ∈ {D4, d4}` — matching the paper's
description precisely. So `trim_flag` *is* the track label:
**`notrim` = simulated replay, `trim` = non-speech removed.** With eval restricted to
the same D4/d4 conditions as a matched reference:

| system | eval D4/d4 (real, w/ non-speech) | → **simulated** | → **no non-speech** |
|---|---|---|---|
| `flatten_T400` | 40.12 | **56.42** (+16.3) | 51.61 (+11.5) |
| `T400` | 39.55 | 54.44 (+14.9) | 53.49 (+13.9) |
| `cmvn_T400` | 43.76 | 58.74 (+15.0) | 49.86 (+6.1) |
| `baseline_T250` | 37.71 | 48.56 (+10.9) | 51.63 (+13.9) |
| `T150` | 36.85 | 44.27 (+7.4) | 53.58 (+16.7) |
| **`flatten_T400_aug1`** | 31.46 | **26.47 (−5.0)** | **30.20 (−1.3)** |
| **`flatten_T400_aug`** | 30.89 | **26.72 (−4.2)** | 32.58 (+1.7) |
| MFCC-SVM | 49.83 | 56.64 (+6.8) | 53.30 (+3.5) |
| MFCC-RF | 41.57 | 44.63 (+3.1) | 37.68 (−3.9) |
| CQCC-GMM *(official)* | 30.02 | 34.94 (+4.9) | 36.65 (+6.6) |
| LFCC-GMM *(official)* | 32.16 | 36.47 (+4.3) | 39.09 (+6.9) |
| LFCC-LCNN *(official)* | 49.02 | 48.98 (−0.0) | 51.65 (+2.6) |
| RawNet2 *(official)* | 43.95 | 45.52 (+1.6) | 44.26 (+0.3) |

**(a) Waveform augmentation INVERTS the simulated-data penalty — under a matched
control.** Every official baseline and every non-augmented system of ours degrades on
simulated replay. Only the augmented models *improve*, and **26.47% is the lowest EER
this project achieves anywhere on 2021** — better than any eval number, and better
than all four baselines on the same subset.

This is not merely consistent with the challenge's own analysis; it is the controlled
version of it. The paper's supplementary §X-C observes: *"not all the CMs performed
worse on the simulated data … One notable difference is that T07, T16, and T04 used
room-impulse-based data augmentation … This suggests that with certain training
strategies, the CMs can do well on both the real and simulated data."* The organisers
could only compare **different teams' entire systems**, which differ in front-end,
backbone, ensembling and augmentation simultaneously. **We have the controlled
experiment**: identical architecture, identical T, identical head, three points on a
single augmentation dose axis, everything else held fixed. That converts a
correlational remark into a causal demonstration — and the paper notes (§V) that
*"post-challenge studies that employ ASVspoof 2021 PA data and work on replay attacks
are rarely seen"*, so this lands in a genuinely under-explored area.

Note also that the paper's own controlled augmentation study (supplementary Table
XIII, on LA/DF) found only *"modest improvements"*, and on 2021 DF the best result
came with **no** augmentation at all. Our PA effect — 7.08 pp, 17.8% relative,
monotonic across three doses, plus this sign inversion — is a stronger demonstration
than the organisers themselves obtained, on the task where it matters most.

**(b) An uncomfortable finding, reported because it is true.** Removing non-speech
costs the official baselines +0.3 to +6.9 pp, but costs our non-augmented models
**+11.5 to +16.7 pp**. Our CQT-LCNNs lean on non-speech regions substantially harder
than the published baselines do. The paper is explicit (§VI, Limitations) that
non-speech *length* "is a database characteristic that should not serve as a cue for
detection", and §IV-A notes CMs relying on non-speech may not detect reliably in the
wild.

Augmentation again largely removes the dependence (aug1 **−1.3**, aug3 +1.7), which is
a third independent argument for it. **Partial confound to state**: VAD-trimmed clips
are shorter, so the tiling factor rises — though `T150` degrading *most* (+16.7),
despite tiling least, argues that tiling is not the whole explanation.

This should be reported as a limitation of the system, not hidden. It also suggests a
concrete follow-up: train on VAD-trimmed audio and see whether the dependence
disappears without costing in-domain accuracy.

### 7.20 Where these systems would have placed in the challenge

Sorting the paper's Table XV evaluation column by EER, `flatten_T400_aug` at
**32.665%** falls between T27 (32.00%) and T28 (32.96%) — **11th of 24** entries
(23 challenge submissions and baselines, plus ours), ahead of 13 including all four
official baselines. The best system in the entire PA track was **T07 at 24.25%**.

**Two caveats, both mandatory whenever this is quoted:**

1. The challenge ranked by **min t-DCF**, not EER (we have not computed t-DCF — see
   `PROJECT_PLAN.md` §9.7), so this is a placement *on EER*, not a challenge placing.
2. Our systems are **not challenge-compliant** on training data — see 7.14.

Context worth carrying into the discussion chapter: the paper's own verdict is that
PA *"appears to be the most challenging of the three"* tasks and that *"the
performance of all systems is substantially worse than the ASV floor of 0.12"*. Every
PA baseline sits at min t-DCF 0.943–1.000, i.e. effectively saturated. A 32.7% EER is
not an outlier failure; it is the neighbourhood the entire field occupies on this
task.

### 7.18 What this phase established

- **Out-of-domain evaluation was not a formality; it reversed the project's
  conclusions.** The best dev system is the second *worst* on 2021. The technique that
  looked monotonically harmful in-domain is the most valuable one out of domain. A
  thesis that reported only dev numbers would have been confidently wrong about both.
- **Pre-registration did real work.** Three of four predictions were adjudicated
  against matched controls, and the fourth was cleanly refuted. Because the list and
  the predictions were fixed in advance, both outcomes are results rather than
  rationalisations — and the one amendment (adding `T400`) is documented with its
  reason and its timing.
- **The thesis's central claim survives the hardest available test**, with the
  front-end advantage holding even without the augmentation win.
- **The pipeline reproduces the published ASVspoof 2021 PA baselines exactly**, on
  three independently-constructed subsets (7.12). Every number in the results chapter
  is anchored to the published record, including the ones no one else has computed.
- **Five of six published condition findings are independently reproduced** (7.15),
  with a different metric and a different system — and two findings the challenge did
  not report are added (attacker-room identity, and the room-group structure).
- **Augmentation reverses the sign of the simulated-data penalty** (7.19) — the
  controlled version of an effect the challenge organisers could only observe
  correlationally across whole systems.
- **The honest headline is dual**: our best system beats every official baseline, and
  it is still at 32.7% EER, which is not a deployable system. Both halves belong in the
  abstract.
- **Naive score fusion does not help** (measured on `progress` only, leaving `eval`
  untouched): mean-z fusion of the three best systems gives 29.668% against 29.795% for
  the best single system — **~0.13 pp**. Recorded so the time is not spent again.

---

**2021 PA eval has now been spent.** Its numbers are a clean generalisation estimate
precisely because nothing was tuned on them. Any future work that selects models using
`eval` retroactively destroys that. The protocol from here is in `PROJECT_PLAN.md`
section 9: develop against `progress`, keep `eval` for a single final confirmation, and
report post-hoc work separately from the pre-registered results.

Next: see `PROJECT_PLAN.md` section 9, "Options moving forward" — a menu, not a plan.
