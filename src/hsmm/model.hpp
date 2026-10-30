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

    // Validates once per construction (O(N^2 + N*D)); the kernels then assume valid parameters.
    Model(int N_, int D_, const double* lpi, const double* lA, const double* lP)
        : N(checked_dim(N_, "N", 2)), D(checked_dim(D_, "D", 1)),
          log_pi(lpi, lpi + N),
          log_A(lA, lA + std::size_t(N) * N),
          log_P(lP, lP + std::size_t(N) * D) {
        validate();
        pi = exp_of(log_pi);
        A = exp_of(log_A);
        P = exp_of(log_P);
    }

private:
    static constexpr double kTol = 1e-6;  // slack for normalisation and for log-probabilities just above 0

    static int checked_dim(int v, const char* name, int min) {
        if (v < min) {
            std::string why = min > 1 ? " (a state cannot transition to itself, so a model needs at least two)" : "";
            throw std::invalid_argument(std::string(name) + " must be >= " + std::to_string(min) + why);
        }
        return v;
    }

    static std::string at(std::size_t i, std::size_t cols) {
        return cols ? "[" + std::to_string(i / cols) + ", " + std::to_string(i % cols) + "]"
                    : "[" + std::to_string(i) + "]";
    }

    // Every entry must be a valid log-probability: not NaN, not +inf, at most 0 (-inf is fine).
    static void check_entries(const std::vector<double>& v, const char* name, std::size_t cols) {
        for (std::size_t i = 0; i < v.size(); ++i)
            if (std::isnan(v[i]) || v[i] > kTol)
                throw std::invalid_argument(std::string(name) + at(i, cols) + " = " + std::to_string(v[i]) +
                                            ", but log-probabilities must be <= 0 (and not NaN)");
    }

    // Each row of width `cols` must sum to 1 as probabilities.
    static void check_rows_normalised(const std::vector<double>& v, const char* name, std::size_t cols) {
        for (std::size_t r = 0; r < v.size() / cols; ++r) {
            double sum = 0;
            for (std::size_t c = 0; c < cols; ++c) sum += std::exp(v[r * cols + c]);
            if (std::abs(sum - 1.0) > kTol)
                throw std::invalid_argument(std::string(name) + (v.size() == cols ? "" : " row " + std::to_string(r)) +
                                            " sums to " + std::to_string(sum) + " as probabilities, expected 1");
        }
    }

    void validate() const {
        check_entries(log_pi, "log_pi", 0);
        check_entries(log_A, "log_A", N);
        check_entries(log_P, "log_P", D);
        for (int i = 0; i < N; ++i)
            if (log_A[std::size_t(i) * N + i] != NEG_INF)
                throw std::invalid_argument("log_A[" + std::to_string(i) + ", " + std::to_string(i) +
                                            "] must be -inf: a state cannot transition to itself");
        check_rows_normalised(log_pi, "log_pi", std::size_t(N));
        check_rows_normalised(log_A, "log_A", std::size_t(N));
        check_rows_normalised(log_P, "log_P", std::size_t(D));
    }

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
