"""Diagonal-covariance GMM with a CHUNKED E-step, for PROJECT_PLAN 9.8c.

    python -m src.gmm --validate    # prove it equals sklearn, then exit

WHY NOT sklearn. `GaussianMixture` materialises the full (n_frames x n_components)
responsibility matrix -- and 2-3 of them live at once, in float64 by default. At the
scale 9.8c needs that is fatal on this machine:

    500,000 frames x 512 components x 8 bytes = 2.05 GB   per array
                                        x 2-3 = 4-6 GB    live, against 5.9 GB total

and it is one contiguous allocation, so it does not degrade gracefully -- it either
fits or raises MemoryError. Closing other applications moves that wall by a fraction
of a GB and does not change how it scales.

WHAT THIS IS, AND IS NOT. Diagonal-covariance EM decomposes into sufficient statistics
that are ADDITIVE over samples:

    N_k = sum_n r_nk        S_k = sum_n r_nk x_n         T_k = sum_n r_nk x_n^2
    then  mu_k = S_k/N_k    var_k = T_k/N_k - mu_k^2     w_k = N_k / sum N

Nothing requires every frame resident at once, so the E-step runs chunk by chunk and
memory is bounded by `chunk x n_components` (~41 MB at chunk=20k, k=512) regardless of
how many frames there are.

**This is exact batch EM computed in pieces -- NOT minibatch or online EM.** Those are
approximations that update parameters from partial data; this accumulates complete
sufficient statistics before every M-step, so it reaches the same fixed point as the
full-batch algorithm and costs nothing statistically. `--validate` is what makes that
claim checkable rather than asserted: given identical initialisation, it must reproduce
sklearn's fitted parameters to numerical tolerance.

Consequence worth stating: frame count is now a RUNTIME decision, not a memory one.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from tqdm import tqdm

from . import config


class DiagGMM:
    """Gaussian mixture, diagonal covariance, fitted by chunked exact EM."""

    def __init__(self, n_components: int = config.GMM_N_COMPONENTS,
                 max_iter: int = config.GMM_MAX_ITER,
                 tol: float = config.GMM_TOL,
                 reg_covar: float = config.GMM_REG_COVAR,
                 chunk: int = config.GMM_CHUNK,
                 seed: int = config.RANDOM_SEED):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.chunk = chunk
        self.seed = seed
        self.weights_ = self.means_ = self.covars_ = None
        self.n_iter_ = 0
        self.lower_bound_ = -np.inf

    # --- initialisation ---------------------------------------------------------

    def _init(self, X: np.ndarray) -> None:
        """Means from random distinct frames, covariances from the global variance.

        This is sklearn's `init_params='random_from_data'` scheme. It is chosen for
        comparability rather than quality: k-means initialisation on ~2 M frames would
        cost more than the EM it precedes, and GMM-UBM practice does not need it.
        """
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=self.k, replace=len(X) < self.k)
        self.means_ = X[idx].astype(np.float64, copy=True)
        var = X.var(axis=0, dtype=np.float64) + self.reg_covar
        self.covars_ = np.tile(var, (self.k, 1))
        self.weights_ = np.full(self.k, 1.0 / self.k)

    # --- E-step -----------------------------------------------------------------

    def _log_prob(self, Xc: np.ndarray) -> np.ndarray:
        """log N(x | mu_k, diag(var_k)) for one chunk -> (n_chunk, k).

        Expanded as sum_d (x-mu)^2/var = x^2 . (1/var) - 2 x . (mu/var) + sum mu^2/var
        so the per-frame work is two matmuls rather than a (n, k, d) broadcast, which
        would be the actual memory blow-up.
        """
        inv = 1.0 / self.covars_                                  # (k, d)
        # log |2*pi*Sigma| and the mu^2/var term are per-component constants
        const = -0.5 * (self.covars_.shape[1] * np.log(2.0 * np.pi)
                        + np.log(self.covars_).sum(axis=1)
                        + (self.means_ ** 2 * inv).sum(axis=1))   # (k,)
        return (const
                - 0.5 * (Xc ** 2) @ inv.T
                + Xc @ (self.means_ * inv).T)                     # (n, k)

    def _e_step_chunk(self, Xc: np.ndarray):
        """Responsibilities for one chunk, plus its total log-likelihood."""
        lw = self._log_prob(Xc) + np.log(self.weights_)
        m = lw.max(axis=1, keepdims=True)
        exp = np.exp(lw - m)
        s = exp.sum(axis=1, keepdims=True)
        return exp / s, float((m[:, 0] + np.log(s[:, 0])).sum())

    # --- fit ----------------------------------------------------------------------

    def fit(self, X: np.ndarray, desc: str = "EM") -> "DiagGMM":
        # Kept at the caller's dtype (float32 in practice) and promoted PER CHUNK.
        # Promoting the whole matrix would cost 864 MB at 1.8 M x 60, which is the
        # kind of quiet doubling this module exists to avoid. Accumulators stay
        # float64, so the sufficient statistics lose nothing.
        X = np.ascontiguousarray(X)
        n, d = X.shape
        self._init(X)
        prev = -np.inf
        bar = tqdm(range(self.max_iter), desc=desc, unit="iter")
        for it in bar:
            N = np.zeros(self.k)
            S = np.zeros((self.k, d))
            T = np.zeros((self.k, d))
            total_ll = 0.0
            for a in range(0, n, self.chunk):          # <-- the whole point
                Xc = X[a:a + self.chunk].astype(np.float64, copy=False)
                r, ll = self._e_step_chunk(Xc)
                total_ll += ll
                N += r.sum(axis=0)
                S += r.T @ Xc
                T += r.T @ (Xc ** 2)
            N = np.maximum(N, 1e-10)                   # a component can empty out
            self.means_ = S / N[:, None]
            self.covars_ = np.maximum(T / N[:, None] - self.means_ ** 2,
                                      0.0) + self.reg_covar
            self.weights_ = N / N.sum()

            mean_ll = total_ll / n
            self.n_iter_, self.lower_bound_ = it + 1, mean_ll
            bar.set_postfix_str(f"logL/frame {mean_ll:.5f}  d {mean_ll - prev:+.2e}")
            if abs(mean_ll - prev) < self.tol:
                bar.close()
                print(f"  converged after {it + 1} iterations "
                      f"(|delta| < {self.tol})", flush=True)
                break
            prev = mean_ll
        return self

    # --- scoring --------------------------------------------------------------------

    def score_frames(self, X: np.ndarray) -> np.ndarray:
        """Per-frame log-likelihood, chunked for the same reason the E-step is."""
        X = np.ascontiguousarray(X)
        out = np.empty(len(X))
        logw = np.log(self.weights_)
        for a in range(0, len(X), self.chunk):
            lw = self._log_prob(X[a:a + self.chunk].astype(np.float64,
                                                           copy=False)) + logw
            m = lw.max(axis=1)
            out[a:a + self.chunk] = m + np.log(np.exp(lw - m[:, None]).sum(axis=1))
        return out

    # --- persistence ------------------------------------------------------------------

    def save(self, path) -> None:
        np.savez_compressed(path, weights=self.weights_, means=self.means_,
                            covars=self.covars_, n_iter=self.n_iter_,
                            lower_bound=self.lower_bound_)

    @classmethod
    def load(cls, path) -> "DiagGMM":
        z = np.load(path)
        g = cls(n_components=len(z["weights"]))
        g.weights_, g.means_, g.covars_ = z["weights"], z["means"], z["covars"]
        g.n_iter_, g.lower_bound_ = int(z["n_iter"]), float(z["lower_bound"])
        return g


def llr(bona: DiagGMM, spoof: DiagGMM, X: np.ndarray) -> float:
    """File score: mean per-frame log-likelihood ratio, bonafide over spoof.

    The MEAN rather than the sum, so a file's score does not scale with its duration.
    That is also why 9.8c samples an equal number of frames per file when fitting --
    training weight then matches the weight each file carries at scoring time.
    """
    return float(np.mean(bona.score_frames(X) - spoof.score_frames(X)))


# --- the declared control -----------------------------------------------------------

def validate(n: int = 50_000, d: int = 20, k: int = 32, seed: int = 0) -> bool:
    """Chunked EM vs sklearn, from IDENTICAL initialisation.

    Declared in 9.8c.1(a). Feeding sklearn our own init is the point: it isolates the
    EM iterations, which is the part actually reimplemented here. Comparing two
    different random starts would only show that both find *a* local optimum, which
    proves nothing about correctness.

    Sized so sklearn can hold its (n x k) float64 responsibilities -- 50k x 32 is
    12.8 MB, against the 2.05 GB the real 500k x 512 fit would need.
    """
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=3.0, size=(6, d))
    X = np.repeat(centres, n // 6 + 1, axis=0)[:n] + rng.normal(size=(n, d))

    ours = DiagGMM(n_components=k, max_iter=25, tol=0.0, reg_covar=1e-6,
                   chunk=4096, seed=seed)
    ours._init(X)
    init_w = ours.weights_.copy()
    init_m = ours.means_.copy()
    init_c = ours.covars_.copy()

    print(f"fitting ours   (n={n:,} d={d} k={k}, chunk={ours.chunk})")
    ours.fit(X, desc="ours")

    print("fitting sklearn (same init, full-batch)")
    sk = GaussianMixture(n_components=k, covariance_type="diag", max_iter=25, tol=0.0,
                         reg_covar=1e-6, weights_init=init_w, means_init=init_m,
                         precisions_init=1.0 / init_c, random_state=seed)
    sk.fit(X)

    # Components can come out in any order only if the init differs; it does not here,
    # so a positional comparison is the strict test.
    dw = np.abs(ours.weights_ - sk.weights_).max()
    dm = np.abs(ours.means_ - sk.means_).max()
    dc = np.abs(ours.covars_ - sk.covariances_).max()
    dl = abs(ours.lower_bound_ - sk.lower_bound_)
    print(f"  max |d weights|   {dw:.3e}")
    print(f"  max |d means|     {dm:.3e}")
    print(f"  max |d covars|    {dc:.3e}")
    print(f"  |d logL/frame|    {dl:.3e}")

    ok = max(dw, dm, dc, dl) < 1e-8
    print("  PASS -- chunked EM is exact batch EM" if ok else
          "  FAIL -- this is not the same algorithm, do not use it")
    return ok


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true",
                   help="check the chunked E-step against sklearn and exit")
    p.add_argument("-n", type=int, default=50_000)
    p.add_argument("-k", type=int, default=32)
    a = p.parse_args()
    if a.validate:
        sys.exit(0 if validate(n=a.n, k=a.k) else 1)
    p.print_help()


if __name__ == "__main__":
    main()
