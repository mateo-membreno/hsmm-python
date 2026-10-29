#pragma once
#include <algorithm>
#include <cstring>
#include <vector>

#include "emission.hpp"
#include "forward.hpp"
#include "model.hpp"

namespace hsmm {

struct EmOptions {
    double var_floor = 1e-6;
    double dur_smoothing = 0.0;  // pseudo-count added to every duration bin before normalising
    bool fit_df = true;          // StudentT: re-estimate degrees of freedom (else keep them fixed)
    double df_min = 2.1, df_max = 300.0;
    bool censored = true;
    std::size_t max_bytes = std::size_t(1) << 31;
};

struct EmResult {
    std::vector<double> log_pi, log_A, log_P, means, vars, df;
    double loglik = 0;  // under the parameters passed in
};

namespace detail {

// Weighted mean / variance with weights wgt[t*N+j] (all ones for Gaussian). Variance is normalised
// by the state's occupancy gsum, which for a t-distribution is the correct scale update.
inline void weighted_mean_var(const Emission& g, const double* Y, std::size_t T, const std::vector<double>& wgt,
                              const std::vector<double>& gsum, const EmOptions& o, EmResult& res) {
    const int N = g.N, K = g.K;
    std::vector<double> sw(N, 0.0), var(std::size_t(N) * K, 0.0);
    std::fill(res.means.begin(), res.means.end(), 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j) {
            sw[j] += wgt[t * N + j];
            for (int k = 0; k < K; ++k) res.means[j * K + k] += wgt[t * N + j] * Y[t * K + k];
        }
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            res.means[j * K + k] = sw[j] > 1e-300 ? res.means[j * K + k] / sw[j] : g.means[j * K + k];
    for (std::size_t t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < K; ++k) {
                const double d = Y[t * K + k] - res.means[j * K + k];
                var[j * K + k] += wgt[t * N + j] * d * d;
            }
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            res.vars[j * K + k] = gsum[j] > 1e-300 ? std::max(var[j * K + k] / gsum[j], o.var_floor)
                                                    : g.vars[j * K + k];
}

inline void gaussian_mstep(const Emission& g, const double* Y, std::size_t T, const std::vector<double>& gamma,
                           const std::vector<double>& gsum, const EmOptions& o, EmResult& res) {
    weighted_mean_var(g, Y, T, gamma, gsum, o, res);
}

// ECM update for a multivariate t with diagonal scale. E-step at the current parameters gives, per
// (t, j), the latent precision weight w = (nu+K)/(nu+delta) and E[log w] = psi((nu+K)/2) - log((nu+delta)/2).
// Location uses gamma*w, scale uses gamma*w over occupancy, and nu solves
//   log(nu/2) + 1 - psi(nu/2) + mean_gamma(E[log w] - w) = 0   (concave in nu, so bisection is exact).
inline void student_t_mstep(const Emission& g, const double* Y, std::size_t T, const std::vector<double>& gamma,
                            const std::vector<double>& gsum, const EmOptions& o, EmResult& res) {
    const int N = g.N, K = g.K;
    std::vector<double> wgt(T * N), tail(N, 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j) {
            const double nu = g.df[j];
            double q = 0;
            for (int k = 0; k < K; ++k) {
                const double d = Y[t * K + k] - g.means[j * K + k];
                q += d * d / g.vars[j * K + k];
            }
            const double w = (nu + K) / (nu + q);
            wgt[t * N + j] = gamma[t * N + j] * w;
            tail[j] += gamma[t * N + j] * (digamma(0.5 * (nu + K)) - std::log(0.5 * (nu + q)) - w);
        }
    weighted_mean_var(g, Y, T, wgt, gsum, o, res);

    res.df.assign(g.df, g.df + N);
    if (!o.fit_df) return;
    for (int j = 0; j < N; ++j) {
        if (gsum[j] <= 1e-300) continue;
        const double c = tail[j] / gsum[j];
        auto h = [&](double nu) { return std::log(0.5 * nu) + 1.0 - digamma(0.5 * nu) + c; };
        double lo = o.df_min, hi = o.df_max;
        if (h(lo) <= 0) { res.df[j] = lo; continue; }
        if (h(hi) >= 0) { res.df[j] = hi; continue; }
        for (int it = 0; it < 80; ++it) {
            const double mid = 0.5 * (lo + hi);
            (h(mid) > 0 ? lo : hi) = mid;
        }
        res.df[j] = 0.5 * (lo + hi);
    }
}

}  // namespace detail

// One Baum-Welch iteration (Gaussian or Student-t emissions, non-parametric durations).
// Forward slices are stored in full ([T*N*D] scaled alphas); the backward pass is a rolling
// [N*D] slice whose statistics are accumulated on the fly. Scaled quantities carry the same
// per-step normalisers c_t, so gamma_t = alpha_t * beta_t needs no extra division.
inline EmResult em_step(const Model& m, const Emission& g, const double* Y, std::size_t T,
                        const EmOptions& o) {
    const int N = m.N, D = m.D, K = g.K;
    const std::size_t ND = std::size_t(N) * D;
    check_budget(8 * (T * ND + T * N * 3 + T), o.max_bytes, "baum-welch forward grid");

    std::vector<double> log_B(T * N), B(T * N), log_m(T), c(T), alpha(T * ND), gamma(T * N);
    emission_log_B(g, Y, T, log_B.data());
    scale_log_B(log_B.data(), T, N, B.data(), log_m.data());

    std::vector<double> in(N);
    double ll = 0;
    for (std::size_t t = 0; t < T; ++t) {
        double* slice = alpha.data() + t * ND;
        if (t > 0) std::memcpy(slice, slice - ND, ND * sizeof(double));
        c[t] = forward_step(m, B.data() + t * N, slice, t == 0, in.data());
        ll += std::log(c[t]) + log_m[t];
    }
    // Non-censored: the last segment must end at T, so terminal beta is 1/tail on r = 0 only.
    double tail = 1.0;
    if (!o.censored) {
        double s = 0;
        for (int j = 0; j < N; ++j) s += alpha[(T - 1) * ND + j * D];
        if (!(s > 0)) throw std::domain_error("no path ends exactly at T");
        tail = s;
        ll += std::log(s);
    }

    std::vector<double> beta(ND), beta_prev(ND), G(N);
    std::vector<double> xi(std::size_t(N) * N, 0.0), dur(std::size_t(N) * D, 0.0), init(N, 0.0);
    if (o.censored) std::fill(beta.begin(), beta.end(), 1.0);
    else for (int j = 0; j < N; ++j) beta[j * D] = 1.0 / tail;

    for (std::size_t t = T; t-- > 0;) {
        const double* a_t = alpha.data() + t * ND;
        const double* b_t = B.data() + t * N;
        const double inv_c = 1.0 / c[t];
        for (int j = 0; j < N; ++j) {
            double gs = 0, gj = 0;
            for (int r = 0; r < D; ++r) {
                gs += a_t[j * D + r] * beta[j * D + r];
                gj += m.P[j * D + r] * beta[j * D + r];
            }
            gamma[t * N + j] = gs;
            G[j] = b_t[j] * gj;  // G_t(k): value of entering k at t
        }
        if (t == 0) {
            for (int k = 0; k < N; ++k) {
                init[k] = m.pi[k] * G[k] * inv_c;
                for (int r = 0; r < D; ++r)
                    dur[k * D + r] += m.pi[k] * m.P[k * D + r] * b_t[k] * beta[k * D + r] * inv_c;
            }
        } else {
            const double* a_p = alpha.data() + (t - 1) * ND;
            for (int k = 0; k < N; ++k) {
                double in_k = 0;
                for (int i = 0; i < N; ++i) {
                    const double x = a_p[i * D] * m.A[i * N + k];
                    in_k += x;
                    xi[i * N + k] += x * G[k] * inv_c;
                }
                for (int r = 0; r < D; ++r)
                    dur[k * D + r] += in_k * m.P[k * D + r] * b_t[k] * beta[k * D + r] * inv_c;
            }
            // beta_{t-1}: shift remaining-time down, or transition out of r = 0.
            for (int j = 0; j < N; ++j) {
                double out = 0;
                for (int k = 0; k < N; ++k) out += m.A[j * N + k] * G[k];
                beta_prev[j * D] = out * inv_c;
                for (int r = 1; r < D; ++r) beta_prev[j * D + r] = b_t[j] * beta[j * D + r - 1] * inv_c;
            }
            std::swap(beta, beta_prev);
        }
    }

    EmResult res;
    res.loglik = ll;
    res.log_pi.assign(m.log_pi.begin(), m.log_pi.end());
    res.log_A.assign(m.log_A.begin(), m.log_A.end());
    res.log_P.assign(m.log_P.begin(), m.log_P.end());
    res.means.assign(g.means, g.means + std::size_t(N) * K);
    res.vars.assign(g.vars, g.vars + std::size_t(N) * K);

    double s = 0;
    for (double v : init) s += v;
    if (s > 0) for (int k = 0; k < N; ++k) res.log_pi[k] = std::log(init[k] / s);

    for (int i = 0; i < N; ++i) {
        double row = 0;
        for (int j = 0; j < N; ++j) row += xi[i * N + j];
        if (row > 0) for (int j = 0; j < N; ++j) res.log_A[i * N + j] = std::log(xi[i * N + j] / row);
    }
    for (int k = 0; k < N; ++k) {
        double tot = 0;
        for (int r = 0; r < D; ++r) tot += dur[k * D + r] + o.dur_smoothing;
        if (tot > 0)
            for (int r = 0; r < D; ++r) res.log_P[k * D + r] = std::log((dur[k * D + r] + o.dur_smoothing) / tot);
    }

    std::vector<double> gsum(N, 0.0);
    for (std::size_t t = 0; t < T; ++t)
        for (int j = 0; j < N; ++j) gsum[j] += gamma[t * N + j];
    switch (g.type) {
        case EmissionType::Gaussian: detail::gaussian_mstep(g, Y, T, gamma, gsum, o, res); break;
        case EmissionType::StudentT: detail::student_t_mstep(g, Y, T, gamma, gsum, o, res); break;
    }
    return res;
}

}  // namespace hsmm
