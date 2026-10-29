from dataclasses import dataclass

import numpy as np

from . import _core


@dataclass
class DecodeResult:
    states: np.ndarray            # [T] MAP state per time step
    segment_states: np.ndarray    # [S]
    segment_durations: np.ndarray  # [S] observed lengths (last one truncated at T)
    final_remaining: int          # steps the last segment is expected to run past T
    log_prob: float


class HSMM:
    """Explicit-duration HSMM with non-parametric duration PMFs.

    emission_type="gaussian": diagonal Gaussian; `vars` are variances.
    emission_type="t": multivariate Student-t with diagonal scale; `vars` are squared scales and
    `df` the per-state degrees of freedom. Pass df=<number> to hold it fixed; otherwise it is
    initialised at 5 and learned in fit().
    """

    def __init__(self, n_states, d_max, emission_type="gaussian", seed=None, df=None):
        if emission_type not in ("gaussian", "t"):
            raise NotImplementedError("emission_type must be 'gaussian' or 't'")
        if df is not None and (emission_type != "t" or not df > 0):
            raise ValueError("df must be positive and only applies to emission_type='t'")
        self.n_states = int(n_states)
        self.d_max = int(d_max)
        self.emission_type = emission_type
        self._fixed_df = df
        self.rng = np.random.default_rng(seed)
        self.log_pi = self.log_A = self.log_P = self.means = self.vars = self.df = None

    def _check(self, Y):
        Y = np.ascontiguousarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2:
            raise ValueError("Y must be [T, K]")
        return Y

    def init_params(self, Y):
        Y = self._check(Y)
        N, D = self.n_states, self.d_max
        idx = self.rng.choice(len(Y), size=N, replace=len(Y) < N)
        self.means = Y[idx].copy()
        self.vars = np.tile(Y.var(axis=0) + 1e-6, (N, 1))
        if self.emission_type == "t":
            self.df = np.full(N, 5.0 if self._fixed_df is None else float(self._fixed_df))
        self.log_pi = np.full(N, -np.log(N))
        A = np.ones((N, N)) - np.eye(N)
        with np.errstate(divide="ignore"):
            self.log_A = np.log(A / A.sum(axis=1, keepdims=True)) if N > 1 else np.zeros((1, 1))
        pmf = _core.discretize_gamma_duration(2.0, max(D / 6.0, 1e-3), D)
        self.log_P = np.tile(np.log(pmf), (N, 1))
        return self

    def log_emissions(self, Y):
        return _core.compute_log_emissions(self._check(Y), self.means, self.vars, self.df)

    def score(self, Y, censored=True):
        return _core.sequence_loglik(self.log_emissions(Y), self.log_pi, self.log_A, self.log_P, censored)

    def decode(self, Y, censored=True, max_bytes=1 << 31):
        s, seg_s, seg_d, rem, lp = _core.viterbi(
            self.log_emissions(Y), self.log_pi, self.log_A, self.log_P, censored, max_bytes
        )
        return DecodeResult(s, seg_s, seg_d, rem, lp)

    def _fit_once(self, Y, n_iter, tol, censored, var_floor, dur_smoothing, max_bytes):
        history = []
        for _ in range(n_iter):
            (self.log_pi, self.log_A, self.log_P, self.means, self.vars, self.df, ll) = _core.em_step(
                Y, self.log_pi, self.log_A, self.log_P, self.means, self.vars, self.df,
                var_floor, dur_smoothing, censored, max_bytes, self._fixed_df is None,
            )
            history.append(ll)
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol * max(1.0, abs(history[-2])):
                break
        return history

    def fit(self, Y, n_iter=50, tol=1e-4, censored=True, var_floor=1e-6, dur_smoothing=1e-3,
            max_bytes=1 << 31, n_restarts=1):
        """Baum-Welch. With n_restarts > 1, EM is run from that many fresh random initialisations
        and the parameters with the best final log-likelihood are kept (warm starts are ignored).
        Returns the per-iteration log-likelihood history of the kept run; per-restart final scores
        are left in `restart_scores_`."""
        Y = self._check(Y)
        if n_restarts < 1:
            raise ValueError("n_restarts must be >= 1")
        args = (n_iter, tol, censored, var_floor, dur_smoothing, max_bytes)
        if n_restarts == 1:
            if self.means is None:
                self.init_params(Y)
            self.restart_scores_ = []
            return self._fit_once(Y, *args)

        best = None
        self.restart_scores_ = []
        for _ in range(n_restarts):
            self.init_params(Y)
            try:
                hist = self._fit_once(Y, *args)
                final = self.score(Y, censored)
            except (ValueError, ArithmeticError, RuntimeError):
                self.restart_scores_.append(float("-inf"))  # degenerate start; skip it
                continue
            self.restart_scores_.append(final)
            if best is None or final > best[0]:
                best = (final, hist, {k: getattr(self, k).copy() for k in self._params()})
        if best is None:
            raise RuntimeError("every restart failed; check the data and d_max")
        for k, v in best[2].items():
            setattr(self, k, v)
        return best[1]

    _FORMAT_VERSION = 1
    _PARAMS = ("log_pi", "log_A", "log_P", "means", "vars")

    def _params(self):
        return self._PARAMS + (("df",) if self.emission_type == "t" else ())

    def save(self, path):
        """Write the fitted parameters to a .npz file (no pickle)."""
        if self.means is None:
            raise ValueError("model has no parameters; call fit() or init_params() first")
        with open(path, "wb") as f:
            np.savez(f, format_version=self._FORMAT_VERSION, emission_type=self.emission_type,
                     fixed_df=np.nan if self._fixed_df is None else self._fixed_df,
                     **{k: getattr(self, k) for k in self._params()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != cls._FORMAT_VERSION:
                raise ValueError(f"unsupported model file version {int(z['format_version'])}")
            etype = str(z["emission_type"]) if "emission_type" in z else "gaussian"
            fixed = float(z["fixed_df"]) if "fixed_df" in z else float("nan")
            keys = cls._PARAMS + (("df",) if etype == "t" else ())
            params = {k: np.ascontiguousarray(z[k], dtype=np.float64) for k in keys}
        N, D = params["log_P"].shape
        K = params["means"].shape[1]
        expected = {"log_pi": (N,), "log_A": (N, N), "log_P": (N, D), "means": (N, K), "vars": (N, K)}
        if etype == "t":
            expected["df"] = (N,)
        for k, shape in expected.items():
            if params[k].shape != shape:
                raise ValueError(f"corrupt model file: {k} has shape {params[k].shape}, expected {shape}")
        model = cls(N, D, emission_type=etype, df=None if np.isnan(fixed) else fixed)
        for k, v in params.items():
            setattr(model, k, v)
        return model

    def filter(self):
        """Streaming filter; call .update(y_t) per tick."""
        return _StreamingFilter(self)

    def remaining_time(self, state, elapsed):
        return _core.remaining_time_distribution(np.exp(self.log_P[state]), elapsed)


class _StreamingFilter:
    def __init__(self, model):
        self.model = model
        self._f = _core.Filter(model.log_pi, model.log_A, model.log_P)

    def update(self, y):
        log_b = self.model.log_emissions(np.atleast_2d(y))[0]
        return self._f.step(log_b) 

    def belief(self):
        return self._f.belief()

    def reset(self):
        self._f.reset()
