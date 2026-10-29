#pragma once
#include <vector>

#include "model.hpp"

namespace hsmm {

namespace detail {
inline std::vector<double> normalised_from_logs(std::vector<double> lp) {
    double m = NEG_INF;
    for (double v : lp) m = std::max(m, v);
    double s = 0;
    for (double& v : lp) { v = std::exp(v - m); s += v; }
    for (double& v : lp) v /= s;
    return lp;
}
}  // namespace detail

// Gamma(shape, scale) density evaluated at d = 1..d_max, renormalised to a PMF.
inline std::vector<double> discretize_gamma_duration(double shape, double scale, int d_max) {
    if (shape <= 0 || scale <= 0 || d_max < 1) throw std::invalid_argument("gamma: need shape, scale > 0 and d_max >= 1");
    std::vector<double> lp(d_max);
    for (int d = 1; d <= d_max; ++d)
        lp[d - 1] = (shape - 1) * std::log(double(d)) - d / scale - std::lgamma(shape) - shape * std::log(scale);
    return detail::normalised_from_logs(std::move(lp));
}

// Poisson(lambda) mass at d = 1..d_max (zero-truncated), renormalised.
inline std::vector<double> discretize_poisson_duration(double lambda, int d_max) {
    if (lambda <= 0 || d_max < 1) throw std::invalid_argument("poisson: need lambda > 0 and d_max >= 1");
    std::vector<double> lp(d_max);
    for (int d = 1; d <= d_max; ++d) lp[d - 1] = d * std::log(lambda) - lambda - std::lgamma(d + 1.0);
    return detail::normalised_from_logs(std::move(lp));
}

// P(Remaining = r | D > e) for r = 1..D-e, from one state's PMF row pmf[D].
inline std::vector<double> remaining_time_distribution(const double* pmf, int D, int elapsed) {
    if (elapsed < 0 || elapsed >= D) throw std::invalid_argument("elapsed must be in [0, D)");
    double tail = 0;
    for (int d = elapsed + 1; d <= D; ++d) tail += pmf[d - 1];
    if (!(tail > 0)) throw std::domain_error("no duration mass beyond elapsed");
    std::vector<double> out(D - elapsed);
    for (int r = 1; r <= D - elapsed; ++r) out[r - 1] = pmf[elapsed + r - 1] / tail;
    return out;
}

}  // namespace hsmm
