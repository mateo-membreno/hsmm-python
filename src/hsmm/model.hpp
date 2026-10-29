#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hsmm {

constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// Explicit-duration model over N states with durations 1..D.
// Layouts (row-major): A[i*N+j] (A_ii = 0), P[i*D+(d-1)].
// Log forms feed Viterbi; linear forms feed the scaled forward/backward passes.
struct Model {
    int N, D;
    std::vector<double> log_pi, log_A, log_P;
    std::vector<double> pi, A, P;

    Model(int N_, int D_, const double* lpi, const double* lA, const double* lP)
        : N(N_), D(D_),
          log_pi(lpi, lpi + N_),
          log_A(lA, lA + std::size_t(N_) * N_),
          log_P(lP, lP + std::size_t(N_) * D_) {
        if (N < 1 || D < 1) throw std::invalid_argument("N and D must be >= 1");
        pi = exp_of(log_pi);
        A = exp_of(log_A);
        P = exp_of(log_P);
    }

private:
    static std::vector<double> exp_of(const std::vector<double>& v) {
        std::vector<double> out(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) out[i] = std::exp(v[i]);
        return out;
    }
};

inline void check_budget(std::size_t bytes, std::size_t max_bytes, const char* what) {
    if (bytes > max_bytes)
        throw std::length_error(std::string(what) + " needs " + std::to_string(bytes >> 20) +
                                " MiB, over max_bytes (" + std::to_string(max_bytes >> 20) +
                                " MiB); use a shorter sequence or smaller d_max");
}

}  // namespace hsmm
