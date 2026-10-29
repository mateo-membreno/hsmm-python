"""Property-based tests: invariants that must hold for any model and any data."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from hsmm import HSMM, Filter, _core, compute_log_emissions, discretize_gamma_duration, discretize_poisson_duration, remaining_time_distribution
from test_hsmm import enumerate_paths, random_model

settings.register_profile("ci", max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("ci")

seeds = st.integers(0, 2**32 - 1)
censor = st.booleans()


@st.composite
def problems(draw, max_N=4, max_D=6, max_T=40, max_K=3):
    """A random valid (log_pi, log_A, log_P, log_B) problem plus its emission inputs."""
    N = draw(st.integers(2, max_N)); D = draw(st.integers(1, max_D))
    T = draw(st.integers(1, max_T)); K = draw(st.integers(1, max_K))
    rng = np.random.default_rng(draw(seeds))
    log_pi, log_A, log_P, means, vars_ = random_model(rng, N, D, K)
    Y = rng.normal(scale=draw(st.floats(0.1, 5.0)), size=(T, K))
    return log_pi, log_A, log_P, means, vars_, Y, compute_log_emissions(Y, means, vars_)


def path_score(log_pi, log_A, log_P, log_B, seg_s, seg_d, rem):
    """Recompute a decoded path's log-probability from its segments."""
    score, t = 0.0, 0
    for k, (s, d) in enumerate(zip(seg_s, seg_d)):
        full = d + (rem if k == len(seg_s) - 1 else 0)
        score += (log_pi[s] if k == 0 else log_A[seg_s[k - 1], s]) + log_P[s, full - 1]
        score += log_B[t:t + d, s].sum()
        t += d
    return score


@given(problems(max_N=3, max_D=3, max_T=5), censor)
def test_loglik_matches_brute_force(p, censored):
    log_pi, log_A, log_P, _, _, _, log_B = p
    scores = [x[0] for x in enumerate_paths(log_pi, log_A, log_P, log_B, censored)]
    want = np.logaddexp.reduce(scores) if scores else -np.inf
    got = _core.sequence_loglik(log_B, log_pi, log_A, log_P, censored) if np.isfinite(want) else None
    if got is not None:
        assert got == pytest.approx(want, abs=1e-9)


@given(problems(), censor)
def test_viterbi_structure(p, censored):
    log_pi, log_A, log_P, _, _, _, log_B = p
    D, T = log_P.shape[1], len(log_B)
    try:
        states, seg_s, seg_d, rem, score = _core.viterbi(log_B, log_pi, log_A, log_P, censored)
    except ValueError:
        pytest.fail("viterbi rejected a valid problem")
    except Exception as e:  # infeasible non-censored endings are allowed to raise
        assert not censored and "feasible" in str(e)
        return
    assert seg_d.sum() == T and (seg_d >= 1).all()
    assert (np.diff(seg_s) != 0).all()                                  # a_ii = 0
    assert (seg_d[:-1] <= D).all() and seg_d[-1] + rem <= D             # duration bound
    assert 0 <= rem < D and (censored or rem == 0)
    assert list(states) == [s for s, d in zip(seg_s, seg_d) for _ in range(d)]
    assert score == pytest.approx(path_score(log_pi, log_A, log_P, log_B, seg_s, seg_d, rem), abs=1e-8)
    assert score <= _core.sequence_loglik(log_B, log_pi, log_A, log_P, censored) + 1e-9


@given(problems())
def test_censored_loglik_dominates(p):
    log_pi, log_A, log_P, _, _, _, log_B = p
    cens = _core.sequence_loglik(log_B, log_pi, log_A, log_P, True)
    try:
        exact = _core.sequence_loglik(log_B, log_pi, log_A, log_P, False)
    except Exception:
        return
    assert exact <= cens + 1e-9


@given(problems())
def test_filter_consistent_with_batch(p):
    log_pi, log_A, log_P, _, _, _, log_B = p
    f, total = Filter(log_pi, log_A, log_P), 0.0
    for row in log_B:
        marg, inc = f.step(row)
        total += inc
        assert (marg >= 0).all() and marg.sum() == pytest.approx(1.0)
    assert total == pytest.approx(_core.sequence_loglik(log_B, log_pi, log_A, log_P, True), abs=1e-8)
    assert f.belief().sum() == pytest.approx(1.0)


@given(problems(), st.floats(-50, 50), censor)
def test_constant_emission_shift(p, c, censored):
    """Adding c to every log-emission shifts loglik by T*c and leaves the MAP path unchanged."""
    log_pi, log_A, log_P, _, _, _, log_B = p
    T = len(log_B)
    try:
        a = _core.viterbi(log_B, log_pi, log_A, log_P, censored)
        b = _core.viterbi(log_B + c, log_pi, log_A, log_P, censored)
        la = _core.sequence_loglik(log_B, log_pi, log_A, log_P, censored)
        lb = _core.sequence_loglik(log_B + c, log_pi, log_A, log_P, censored)
    except Exception:
        return
    assert lb == pytest.approx(la + T * c, abs=1e-7)
    assert b[4] == pytest.approx(a[4] + T * c, abs=1e-7)
    assert (a[0] == b[0]).all()


@given(problems(), seeds, censor)
def test_state_relabelling_equivariance(p, seed, censored):
    log_pi, log_A, log_P, _, _, _, log_B = p
    perm = np.random.default_rng(seed).permutation(len(log_pi))
    args = (log_B[:, perm], log_pi[perm], log_A[np.ix_(perm, perm)], log_P[perm], censored)
    try:
        base = _core.viterbi(log_B, log_pi, log_A, log_P, censored)
        perm_res = _core.viterbi(*args)
    except Exception:
        return
    assert _core.sequence_loglik(*args) == pytest.approx(_core.sequence_loglik(log_B, log_pi, log_A, log_P, censored), abs=1e-8)
    assert perm_res[4] == pytest.approx(base[4], abs=1e-8)


@given(st.floats(0.2, 20), st.floats(0.1, 20), st.integers(1, 80))
def test_gamma_pmf_is_distribution(shape, scale, d_max):
    pmf = discretize_gamma_duration(shape, scale, d_max)
    assert pmf.shape == (d_max,) and (pmf >= 0).all() and pmf.sum() == pytest.approx(1.0)


@given(st.floats(0.1, 30), st.integers(1, 80))
def test_poisson_pmf_is_distribution(lam, d_max):
    pmf = discretize_poisson_duration(lam, d_max)
    assert (pmf >= 0).all() and pmf.sum() == pytest.approx(1.0)


@given(st.floats(0.2, 10), st.floats(0.5, 10), st.integers(2, 60), st.data())
def test_remaining_time_is_conditional_pmf(shape, scale, d_max, data):
    pmf = discretize_gamma_duration(shape, scale, d_max)
    e = data.draw(st.integers(0, d_max - 1))
    if pmf[e:].sum() <= 0:
        return
    rem = remaining_time_distribution(pmf, e)
    assert rem.shape == (d_max - e,) and rem.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(rem, pmf[e:] / pmf[e:].sum())


@settings(max_examples=15)
@given(seeds, st.integers(2, 3), st.integers(2, 8), st.integers(30, 120), censor)
def test_em_never_decreases_likelihood(seed, N, D, T, censored):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(T, 2)) + rng.integers(0, 3, size=(T, 1)) * 3.0
    model = HSMM(N, D, seed=seed)
    hist = model.fit(Y, n_iter=8, tol=0, censored=censored, dur_smoothing=0.0)
    assert all(b >= a - 1e-7 * max(1, abs(a)) for a, b in zip(hist, hist[1:]))
    # parameters stay valid distributions
    assert np.exp(model.log_pi).sum() == pytest.approx(1.0)
    assert np.exp(model.log_A).sum(1) == pytest.approx(np.ones(N))
    assert (np.diag(model.log_A) == -np.inf).all()
    assert np.exp(model.log_P).sum(1) == pytest.approx(np.ones(N))
    assert (model.vars > 0).all()


@settings(max_examples=15)
@given(seeds, st.integers(2, 3), st.integers(2, 8), st.integers(40, 120), censor, st.booleans())
def test_t_em_never_decreases_likelihood(seed, N, D, T, censored, learn_df):
    rng = np.random.default_rng(seed)
    Y = rng.standard_t(3, size=(T, 2)) + rng.integers(0, 3, size=(T, 1)) * 3.0
    model = HSMM(N, D, emission_type="t", seed=seed, df=None if learn_df else 4.0)
    hist = model.fit(Y, n_iter=8, tol=0, censored=censored, dur_smoothing=0.0)
    assert all(b >= a - 1e-7 * max(1, abs(a)) for a, b in zip(hist, hist[1:]))
    assert (model.vars > 0).all() and (model.df >= 2.1 - 1e-12).all() and (model.df <= 300 + 1e-9).all()
    if not learn_df:
        assert (model.df == 4.0).all()
