#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "model.hpp"

namespace hsmm {

struct ViterbiResult {
    std::vector<int> path;           // state at each t
    std::vector<int> seg_states;     // state of each segment
    std::vector<int> seg_durations;  // observed length of each segment (last one is truncated by T)
    int final_remaining = 0;         // steps the last segment would still run past T
    double score = NEG_INF;          // log-probability of the MAP path
};

// Exact MAP decoding in the remaining-duration form. Only the rolling delta slice is kept; the
// trellis is stored as one byte per (t,j,r) (continue vs. enter) plus the best entering predecessor
// per (t,j), so memory is T*N*(D+4) bytes.
inline ViterbiResult viterbi(const Model& m, const double* log_B, std::size_t T, bool censored,
                             std::size_t max_bytes) {
    const int N = m.N, D = m.D;
    const std::size_t ND = std::size_t(N) * D;
    check_budget(T * N * (D + sizeof(int)), max_bytes, "viterbi trellis");
    std::vector<std::uint8_t> enter(T * ND);
    std::vector<int> best_pred(T * N);
    std::vector<double> cur(ND), nxt(ND), bt(N);

    for (int j = 0; j < N; ++j)
        for (int r = 0; r < D; ++r) cur[j * D + r] = m.log_pi[j] + m.log_P[j * D + r] + log_B[j];

    for (std::size_t t = 1; t < T; ++t) {
        for (int j = 0; j < N; ++j) {
            double best = NEG_INF;
            int bi = 0;
            for (int i = 0; i < N; ++i) {
                const double v = cur[i * D] + m.log_A[i * N + j];
                if (v > best) { best = v; bi = i; }
            }
            bt[j] = best;
            best_pred[t * N + j] = bi;
        }
        for (int j = 0; j < N; ++j)
            for (int r = 0; r < D; ++r) {
                const double cont = (r + 1 < D) ? cur[j * D + r + 1] : NEG_INF;
                const double tr = bt[j] + m.log_P[j * D + r];
                const bool e = tr > cont;
                nxt[j * D + r] = (e ? tr : cont) + log_B[t * N + j];
                enter[t * ND + j * D + r] = e;
            }
        std::swap(cur, nxt);
    }

    ViterbiResult res;
    int bj = -1, br = 0;
    for (int j = 0; j < N; ++j)
        for (int r = 0; r < (censored ? D : 1); ++r)
            if (cur[j * D + r] > res.score) { res.score = cur[j * D + r]; bj = j; br = r; }
    if (bj < 0) throw std::domain_error("no feasible path (score is -inf)");

    res.final_remaining = br;
    res.path.assign(T, 0);
    int j = bj, r = br;
    std::size_t t = T - 1, end = T;
    res.path[t] = j;
    while (t > 0) {
        if (enter[t * ND + j * D + r]) {
            res.seg_states.push_back(j);
            res.seg_durations.push_back(int(end - t));
            end = t;
            j = best_pred[t * N + j];
            r = 0;
        } else {
            ++r;
        }
        --t;
        res.path[t] = j;
    }
    res.seg_states.push_back(j);
    res.seg_durations.push_back(int(end));
    std::reverse(res.seg_states.begin(), res.seg_states.end());
    std::reverse(res.seg_durations.begin(), res.seg_durations.end());
    return res;
}

}  // namespace hsmm
