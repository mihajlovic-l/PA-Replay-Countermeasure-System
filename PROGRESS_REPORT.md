# Progress Report — Phases 0 through 3

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

Next: **Phase 4**, DSP feature extraction (MFCC + CQT), per `PROJECT_PLAN.md` section 6.
