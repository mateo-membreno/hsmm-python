import itertools

import numpy as np
import pytest

from hsmm import HSMM, Filter, _core, compute_log_emissions, discretize_gamma_duration, remaining_time_distribution


def random_model(rng, N=3, D=4, K=2):
    A = rng.random((N, N)); np.fill_diagonal(A, 0); A /= A.sum(1, keepdims=True)
    P = rng.random((N, D)) + 0.05; P /= P.sum(1, keepdims=True)
    pi = rng.random(N) + 0.1; pi /= pi.sum()
    with np.errstate(divide="ignore"):
        return np.log(pi), np.log(A), np.log(P), rng.normal(size=(N, K)), rng.random((N, K)) + 0.3


def enumerate_paths(log_pi, log_A, log_P, log_B, censored):
    """Yield (score, states, durations, final_remaining) for every full segmentation."""
    T, N = log_B.shape
    D = log_P.shape[1]

    def rec(t, prev, score, segs):
        for s in range(N):
            if prev is not None and s == prev:
                continue
            head = log_pi[s] if prev is None else log_A[prev, s]
            for d in range(1, D + 1):
                end = min(t + d, T)
                sc = score + head + log_P[s, d - 1] + log_B[t:end, s].sum()
                if t + d < T:
                    yield from rec(t + d, s, sc, segs + [(s, d)])
                elif t + d == T or censored:
                    yield sc, segs + [(s, end - t)], t + d - T

    for sc, segs, rem in rec(0, None, 0.0, []):
        yield sc, segs, rem


@pytest.mark.parametrize("censored", [True, False])
def test_likelihood_and_viterbi_match_brute_force(censored):
    rng = np.random.default_rng(0)
    log_pi, log_A, log_P, means, vars_ = random_model(rng)
    Y = rng.normal(size=(6, 2))
    log_B = compute_log_emissions(Y, means, vars_)
    paths = list(enumerate_paths(log_pi, log_A, log_P, log_B, censored))
    scores = np.array([p[0] for p in paths])
    expect_ll = np.logaddexp.reduce(scores)
    assert _core.sequence_loglik(log_B, log_pi, log_A, log_P, censored) == pytest.approx(expect_ll)

    states, seg_s, seg_d, rem, score = _core.viterbi(log_B, log_pi, log_A, log_P, censored)
    best = int(np.argmax(scores))
    assert score == pytest.approx(scores[best])
    assert list(zip(seg_s, seg_d)) == paths[best][1]
    assert rem == paths[best][2]
    assert list(states) == [s for s, d in paths[best][1] for _ in range(d)]


def test_emissions_match_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(10, 3)); means = rng.normal(size=(4, 3)); vars_ = rng.random((4, 3)) + 0.2
    got = compute_log_emissions(Y, means, vars_)
    for i in range(4):
        want = scipy_stats.multivariate_normal(means[i], np.diag(vars_[i])).logpdf(Y)
        np.testing.assert_allclose(got[:, i], want)


def test_filter_matches_batch_and_posterior():
    rng = np.random.default_rng(2)
    log_pi, log_A, log_P, means, vars_ = random_model(rng)
    Y = rng.normal(size=(7, 2))
    log_B = compute_log_emissions(Y, means, vars_)
    f = Filter(log_pi, log_A, log_P)
    total = 0.0
    for t in range(len(Y)):
        marg, inc = f.step(log_B[t])
        total += inc
        assert marg.sum() == pytest.approx(1.0)
    assert total == pytest.approx(_core.sequence_loglik(log_B, log_pi, log_A, log_P, True))

    # filtered marginal at T-1 vs brute-force posterior over the state at the last step
    paths = list(enumerate_paths(log_pi, log_A, log_P, log_B, True))
    w = np.array([p[0] for p in paths]); w = np.exp(w - w.max()); w /= w.sum()
    post = np.zeros(3)
    for wi, p in zip(w, paths):
        post[p[1][-1][0]] += wi
    np.testing.assert_allclose(marg, post)
    assert f.belief().shape == log_P.shape and f.belief().sum() == pytest.approx(1.0)
    f.reset()
    marg2, _ = f.step(log_B[0])
    assert marg2.sum() == pytest.approx(1.0)


def test_duration_helpers():
    pmf = discretize_gamma_duration(3.0, 2.0, 30)
    assert pmf.sum() == pytest.approx(1.0) and pmf.argmax() + 1 in (4, 5)
    rem = remaining_time_distribution(pmf, 5)
    assert rem.shape == (25,) and rem.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(rem[0], pmf[5] / pmf[5:].sum())
    with pytest.raises(ValueError):
        remaining_time_distribution(pmf, 30)


def simulate(rng, T, means, stds, dur_means):
    states, ys, s = [], [], 0
    while len(states) < T:
        d = 1 + rng.poisson(dur_means[s])
        states += [s] * d
        s = 1 - s
    states = np.array(states[:T])
    return states, rng.normal(means[states], stds[states])[:, None]


@pytest.mark.parametrize("censored", [True, False])
def test_em_monotone_and_recovers_regimes(censored):
    rng = np.random.default_rng(3)
    true_states, Y = simulate(rng, 600, np.array([-2.0, 2.0]), np.array([0.5, 0.5]), [8, 15])
    model = HSMM(2, 30, seed=0)
    hist = model.fit(Y, n_iter=40, tol=0, censored=censored)
    assert all(b >= a - 1e-6 for a, b in zip(hist, hist[1:]))
    assert hist[-1] > hist[0]
    assert sorted(np.round(model.means.ravel(), 0)) == [-2.0, 2.0]
    dec = model.decode(Y, censored=censored)
    acc = max((dec.states == true_states).mean(), (dec.states != true_states).mean())
    assert acc > 0.95
    assert dec.segment_durations.sum() == len(Y)


def test_memory_guard_and_validation():
    rng = np.random.default_rng(4)
    log_pi, log_A, log_P, means, vars_ = random_model(rng)
    log_B = compute_log_emissions(rng.normal(size=(50, 2)), means, vars_)
    with pytest.raises(ValueError, match="max_bytes"):
        _core.viterbi(log_B, log_pi, log_A, log_P, True, 100)
    with pytest.raises(ValueError):
        _core.viterbi(log_B[:, :2], log_pi, log_A, log_P, True, 1 << 30)
    with pytest.raises(ValueError):
        compute_log_emissions(rng.normal(size=(5, 2)), means, np.zeros_like(vars_))
    with pytest.raises(NotImplementedError):
        HSMM(2, 5, emission_type="poisson")


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    _, Y = simulate(rng, 300, np.array([-2.0, 2.0]), np.array([0.5, 0.5]), [8, 15])
    model = HSMM(2, 25, seed=0)
    model.fit(Y, n_iter=10)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = HSMM.load(path)
    assert (loaded.n_states, loaded.d_max) == (2, 25)
    for k in HSMM._PARAMS:
        np.testing.assert_array_equal(getattr(loaded, k), getattr(model, k))
    assert loaded.score(Y) == model.score(Y)
    a, b = model.decode(Y), loaded.decode(Y)
    np.testing.assert_array_equal(a.states, b.states)
    f1, f2 = model.filter(), loaded.filter()
    for y in Y[:20]:
        np.testing.assert_array_equal(f1.update(y)[0], f2.update(y)[0])


def test_save_load_errors(tmp_path):
    with pytest.raises(ValueError, match="no parameters"):
        HSMM(2, 5).save(tmp_path / "x.npz")
    bad = tmp_path / "bad.npz"
    np.savez(bad, format_version=99)
    with pytest.raises(ValueError, match="version"):
        HSMM.load(bad)
    m = HSMM(2, 5, seed=0).init_params(np.random.default_rng(0).normal(size=(20, 2)))
    m.save(tmp_path / "ok.npz")
    z = dict(np.load(tmp_path / "ok.npz"))
    z["log_A"] = z["log_A"][:1]
    np.savez(bad, **z)
    with pytest.raises(ValueError, match="corrupt"):
        HSMM.load(bad)


def test_restarts_never_worse_than_single_fit():
    rng = np.random.default_rng(6)
    _, Y = simulate(rng, 400, np.array([-2.0, 2.0]), np.array([0.7, 0.7]), [6, 12])
    for seed in range(5):
        single = HSMM(3, 20, seed=seed)
        single.fit(Y, n_iter=30)
        multi = HSMM(3, 20, seed=seed)
        multi.fit(Y, n_iter=30, n_restarts=6)
        assert len(multi.restart_scores_) == 6
        # the first restart shares the single fit's random start, so best-of can't lose to it
        assert multi.score(Y) >= single.score(Y) - 1e-9
        assert multi.score(Y) == pytest.approx(max(multi.restart_scores_))
    with pytest.raises(ValueError):
        HSMM(2, 5).fit(Y, n_restarts=0)


def test_t_emissions_match_scipy_and_gaussian_limit():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(12, 3)); means = rng.normal(size=(3, 3)); scales2 = rng.random((3, 3)) + 0.3
    df = np.array([2.5, 5.0, 30.0])
    got = compute_log_emissions(Y, means, scales2, df)
    for i in range(3):
        want = scipy_stats.multivariate_t(means[i], np.diag(scales2[i]), df=df[i]).logpdf(Y)
        np.testing.assert_allclose(got[:, i], want, rtol=1e-10)
    big = compute_log_emissions(Y, means, scales2, np.full(3, 1e9))
    np.testing.assert_allclose(big, compute_log_emissions(Y, means, scales2), atol=1e-5)
    with pytest.raises(ValueError):
        compute_log_emissions(Y, means, scales2, np.array([5.0, -1.0, 5.0]))
    with pytest.raises(ValueError):
        compute_log_emissions(Y, means, scales2, np.array([5.0, 5.0]))


def simulate_t(rng, T, means, scales, nu, dur_means):
    states, s = [], 0
    while len(states) < T:
        states += [s] * (1 + rng.poisson(dur_means[s]))
        s = 1 - s
    states = np.array(states[:T])
    return states, (means[states] + scales[states] * rng.standard_t(nu, size=T))[:, None]


@pytest.mark.parametrize("censored", [True, False])
def test_t_em_monotone_and_learns_df(censored):
    rng = np.random.default_rng(8)
    true_states, Y = simulate_t(rng, 3000, np.array([-1.0, 1.0]), np.array([0.4, 0.4]), 3.0, [10, 15])
    model = HSMM(2, 40, emission_type="t", seed=0)
    hist = model.fit(Y, n_iter=60, tol=0, censored=censored, n_restarts=3)
    assert all(b >= a - 1e-6 * max(1, abs(a)) for a, b in zip(hist, hist[1:]))
    assert sorted(np.round(model.means.ravel())) == [-1.0, 1.0]
    assert (model.df > 2.1).all() and (model.df < 8).all()      # true nu = 3, Gaussian would be >> 8
    dec = model.decode(Y, censored=censored)
    assert max((dec.states == true_states).mean(), (dec.states != true_states).mean()) > 0.9


def test_t_beats_gaussian_on_heavy_tails_and_fixed_df_is_held():
    rng = np.random.default_rng(9)
    _, Y = simulate_t(rng, 2000, np.array([-1.0, 1.0]), np.array([0.4, 0.4]), 3.0, [10, 15])
    g = HSMM(2, 40, seed=0); g.fit(Y, n_iter=60, n_restarts=3)
    t = HSMM(2, 40, emission_type="t", seed=0); t.fit(Y, n_iter=60, n_restarts=3)
    assert t.score(Y) > g.score(Y) + 20
    fixed = HSMM(2, 40, emission_type="t", seed=0, df=4.0); fixed.fit(Y, n_iter=20)
    np.testing.assert_array_equal(fixed.df, [4.0, 4.0])
    with pytest.raises(ValueError):
        HSMM(2, 5, df=4.0)                      # df only applies to the t emission
    with pytest.raises(ValueError):
        HSMM(2, 5, emission_type="t", df=-1)


def test_t_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(10)
    _, Y = simulate_t(rng, 400, np.array([-1.0, 1.0]), np.array([0.4, 0.4]), 4.0, [8, 12])
    for fixed in (None, 6.0):
        model = HSMM(2, 25, emission_type="t", seed=0, df=fixed)
        model.fit(Y, n_iter=10)
        model.save(tmp_path / "t.npz")
        loaded = HSMM.load(tmp_path / "t.npz")
        assert loaded.emission_type == "t" and loaded._fixed_df == fixed
        np.testing.assert_array_equal(loaded.df, model.df)
        assert loaded.score(Y) == model.score(Y)
        np.testing.assert_array_equal(loaded.decode(Y).states, model.decode(Y).states)
        f1, f2 = model.filter(), loaded.filter()
        for y in Y[:10]:
            np.testing.assert_array_equal(f1.update(y)[0], f2.update(y)[0])
