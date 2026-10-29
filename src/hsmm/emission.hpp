#pragma once
#include <algorithm>
#include <vector>

#include "model.hpp"

namespace hsmm {

enum class EmissionType { Gaussian, StudentT };

// Per-state parameters, flat row-major [N*K]. For StudentT, `vars` holds the squared scale of a
// multivariate t with diagonal scale matrix, and `df` ([N], null for Gaussian) the degrees of freedom.
struct Emission {
    EmissionType type;
    int N, K;
    const double* means;
    const double* vars;
    const double* df;
};

inline double digamma(double x) {
    double r = 0;
    while (x < 6) { r -= 1.0 / x; x += 1.0; }
    const double f = 1.0 / (x * x);
    return r + std::log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
}

namespace detail {

inline void gaussian_log_B(const Emission& g, const double* Y, std::size_t T, double* out) {
    constexpr double kTwoPi = 6.283185307179586;
    const int N = g.N, K = g.K;
    std::vector<double> norm(N);
    for (int i = 0; i < N; ++i) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += std::log(kTwoPi * g.vars[i * K + k]);
        norm[i] = -0.5 * s;
    }
    for (std::size_t t = 0; t < T; ++t) {
        const double* y = Y + t * K;
        for (int i = 0; i < N; ++i) {
            const double* mu = g.means + i * K;
            const double* var = g.vars + i * K;
            double q = 0;
            for (int k = 0; k < K; ++k) {
                const double d = y[k] - mu[k];
                q += d * d / var[k];
            }
            out[t * N + i] = norm[i] - 0.5 * q;
        }
    }
}

inline void student_t_log_B(const Emission& g, const double* Y, std::size_t T, double* out) {
    constexpr double kPi = 3.141592653589793;
    const int N = g.N, K = g.K;
    std::vector<double> norm(N);
    for (int i = 0; i < N; ++i) {
        const double nu = g.df[i];
        double s = 0;
        for (int k = 0; k < K; ++k) s += std::log(g.vars[i * K + k]);
        norm[i] = std::lgamma(0.5 * (nu + K)) - std::lgamma(0.5 * nu) - 0.5 * K * std::log(nu * kPi) - 0.5 * s;
    }
    for (std::size_t t = 0; t < T; ++t) {
        const double* y = Y + t * K;
        for (int i = 0; i < N; ++i) {
            const double nu = g.df[i];
            const double* mu = g.means + i * K;
            const double* var = g.vars + i * K;
            double q = 0;
            for (int k = 0; k < K; ++k) {
                const double d = y[k] - mu[k];
                q += d * d / var[k];
            }
            out[t * N + i] = norm[i] - 0.5 * (nu + K) * std::log1p(q / nu);
        }
    }
}

}  // namespace detail

// Fill out[t*N+i] = log p(Y[t] | state i). The type switch sits outside the T x N loops.
inline void emission_log_B(const Emission& g, const double* Y, std::size_t T, double* out) {
    switch (g.type) {
        case EmissionType::Gaussian: detail::gaussian_log_B(g, Y, T, out); return;
        case EmissionType::StudentT: detail::student_t_log_B(g, Y, T, out); return;
    }
}

// Per-timestep max-shifted exponentials: B[t*N+i] = exp(log_B - m_t), log_m[t] = m_t.
inline void scale_log_B(const double* log_B, std::size_t T, int N, double* B, double* log_m) {
    for (std::size_t t = 0; t < T; ++t) {
        double m = NEG_INF;
        for (int i = 0; i < N; ++i) m = std::max(m, log_B[t * N + i]);
        if (!std::isfinite(m)) throw std::domain_error("every state has zero emission likelihood at some t");
        log_m[t] = m;
        for (int i = 0; i < N; ++i) B[t * N + i] = std::exp(log_B[t * N + i] - m);
    }
}

}  // namespace hsmm
