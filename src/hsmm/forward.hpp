#pragma once
#include <algorithm>
#include <cstring>
#include <vector>

#include "emission.hpp"
#include "model.hpp"

namespace hsmm {

// Remaining-duration forward recursion. alpha[j*D+r] = P(y_1..t, state j at t, r steps left after t),
// r = 0 meaning the segment ends at t:
//   alpha_t(j,r) = [alpha_{t-1}(j,r+1) + (sum_i alpha_{t-1}(i,0) A_ij) * P_j(r+1)] * b_j(t)
// One step costs O(N^2 + N*D). Updates alpha in place (no allocation; `in` is an N-sized scratch),
// normalises it to sum to 1 and returns the normaliser. `b` is the max-shifted linear emission row.
inline double forward_step(const Model& m, const double* b, double* alpha, bool first, double* in) {
    const int N = m.N, D = m.D;
    double c = 0;
    if (first) {
        for (int j = 0; j < N; ++j)
            for (int r = 0; r < D; ++r) {
                const double a = m.pi[j] * m.P[j * D + r] * b[j];
                alpha[j * D + r] = a;
                c += a;
            }
    } else {
        std::fill(in, in + N, 0.0);
        for (int i = 0; i < N; ++i) {
            const double a0 = alpha[i * D];
            if (a0 == 0) continue;
            for (int j = 0; j < N; ++j) in[j] += a0 * m.A[i * N + j];
        }
        for (int j = 0; j < N; ++j) {
            double* aj = alpha + j * D;
            const double* pj = m.P.data() + j * D;
            const double bj = b[j], inj = in[j];
            for (int r = 0; r < D; ++r) {  // ascending r reads aj[r+1] before it is overwritten
                const double cont = (r + 1 < D) ? aj[r + 1] : 0.0;
                const double a = (cont + inj * pj[r]) * bj;
                aj[r] = a;
                c += a;
            }
        }
    }
    if (!(c > 0)) throw std::domain_error("forward pass hit zero probability (model cannot explain the data)");
    const double inv = 1.0 / c;
    for (int k = 0; k < N * D; ++k) alpha[k] *= inv;
    return c;
}

// log P(Y | model). censored: the last segment may run past T; otherwise it must end exactly at T.
inline double sequence_loglik(const Model& m, const double* log_B, std::size_t T, bool censored) {
    const int N = m.N, D = m.D;
    std::vector<double> alpha(std::size_t(N) * D), in(N), b(N);
    double ll = 0;
    for (std::size_t t = 0; t < T; ++t) {
        double mx = NEG_INF;
        for (int i = 0; i < N; ++i) mx = std::max(mx, log_B[t * N + i]);
        if (!std::isfinite(mx)) throw std::domain_error("every state has zero emission likelihood at some t");
        for (int i = 0; i < N; ++i) b[i] = std::exp(log_B[t * N + i] - mx);
        ll += std::log(forward_step(m, b.data(), alpha.data(), t == 0, in.data())) + mx;
    }
    if (!censored) {
        double tail = 0;
        for (int j = 0; j < N; ++j) tail += alpha[j * D];
        ll += std::log(tail);  // -inf if no path ends exactly at T
    }
    return ll;
}

// Streaming belief state P(S_t, remaining | y_1..t); step() does no heap allocation.
class Filter {
public:
    explicit Filter(Model model)
        : m_(std::move(model)), alpha_(std::size_t(m_.N) * m_.D), in_(m_.N), b_(m_.N) {}

    void reset() { first_ = true; }

    // Consume log P(y_t | state i) for all i. Writes P(S_t = i | y_1..t) to marginals[N];
    // returns log P(y_t | y_1..t-1).
    double step(const double* log_b, double* marginals) {
        const int N = m_.N, D = m_.D;
        double mx = NEG_INF;
        for (int i = 0; i < N; ++i) mx = std::max(mx, log_b[i]);
        if (!std::isfinite(mx)) throw std::domain_error("every state has zero emission likelihood");
        for (int i = 0; i < N; ++i) b_[i] = std::exp(log_b[i] - mx);
        const double c = forward_step(m_, b_.data(), alpha_.data(), first_, in_.data());
        first_ = false;
        for (int j = 0; j < N; ++j) {
            double s = 0;
            for (int r = 0; r < D; ++r) s += alpha_[j * D + r];
            marginals[j] = s;
        }
        return std::log(c) + mx;
    }

    // Joint belief [N*D]; entry (j, r) = P(state j, r steps remaining).
    const std::vector<double>& alpha() const { return alpha_; }
    const Model& model() const { return m_; }

private:
    Model m_;
    std::vector<double> alpha_, in_, b_;
    bool first_ = true;
};

}  // namespace hsmm
