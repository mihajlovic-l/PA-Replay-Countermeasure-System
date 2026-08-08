# Progress Report — Phases 0 through 5

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

Next: **Phase 6**, the CQT-LCNN main system, per `PROJECT_PLAN.md` section 6.
