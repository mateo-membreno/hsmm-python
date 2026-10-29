#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <initializer_list>
#include <optional>
#include <vector>

#include "hsmm/durations.hpp"
#include "hsmm/em.hpp"
#include "hsmm/emission.hpp"
#include "hsmm/forward.hpp"
#include "hsmm/model.hpp"
#include "hsmm/viterbi.hpp"

namespace nb = nanobind;
using namespace nb::literals;

using Arr1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Arr2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

namespace {

template <typename T>
nb::ndarray<nb::numpy, T> wrap(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    return nb::ndarray<nb::numpy, T>(p->data(), shape.size(), shape.begin(), owner);
}

hsmm::Model make_model(Arr1 log_pi, Arr2 log_A, Arr2 log_P) {
    const std::size_t N = log_pi.shape(0);
    if (log_A.shape(0) != N || log_A.shape(1) != N) throw std::invalid_argument("log_A must be [N, N]");
    if (log_P.shape(0) != N) throw std::invalid_argument("log_P must be [N, D_max]");
    return hsmm::Model(int(N), int(log_P.shape(1)), log_pi.data(), log_A.data(), log_P.data());
}

void check_log_B(const hsmm::Model& m, Arr2 log_B) {
    if (log_B.shape(1) != std::size_t(m.N)) throw std::invalid_argument("log_B must be [T, N]");
    if (log_B.shape(0) < 1) throw std::invalid_argument("empty sequence");
}

// df given -> multivariate Student-t (vars are squared scales); df absent -> Gaussian.
hsmm::Emission make_emission(Arr2 means, Arr2 vars, const std::optional<Arr1>& df) {
    if (means.shape(0) != vars.shape(0) || means.shape(1) != vars.shape(1))
        throw std::invalid_argument("means and vars must have the same [N, K] shape");
    for (std::size_t i = 0; i < vars.size(); ++i)
        if (!(vars.data()[i] > 0)) throw std::invalid_argument("vars must be positive");
    hsmm::Emission e{hsmm::EmissionType::Gaussian, int(means.shape(0)), int(means.shape(1)),
                     means.data(), vars.data(), nullptr};
    if (df) {
        if (df->shape(0) != means.shape(0)) throw std::invalid_argument("df must have length N");
        for (std::size_t i = 0; i < df->size(); ++i)
            if (!(df->data()[i] > 0)) throw std::invalid_argument("df must be positive");
        e.type = hsmm::EmissionType::StudentT;
        e.df = df->data();
    }
    return e;
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Explicit-duration HSMM engine";

    m.def("discretize_gamma_duration", [](double shape, double scale, int d_max) {
        return wrap(hsmm::discretize_gamma_duration(shape, scale, d_max), {std::size_t(d_max)});
    }, "shape"_a, "scale"_a, "d_max"_a);

    m.def("discretize_poisson_duration", [](double lam, int d_max) {
        return wrap(hsmm::discretize_poisson_duration(lam, d_max), {std::size_t(d_max)});
    }, "lam"_a, "d_max"_a);

    m.def("remaining_time_distribution", [](Arr1 pmf, int elapsed) {
        auto out = hsmm::remaining_time_distribution(pmf.data(), int(pmf.shape(0)), elapsed);
        const std::size_t n = out.size();
        return wrap(std::move(out), {n});
    }, "pmf"_a, "elapsed"_a);

    m.def("compute_log_emissions", [](Arr2 Y, Arr2 means, Arr2 vars, const std::optional<Arr1>& df) {
        auto g = make_emission(means, vars, df);
        if (Y.shape(1) != std::size_t(g.K)) throw std::invalid_argument("Y must be [T, K]");
        const std::size_t T = Y.shape(0);
        std::vector<double> out(T * g.N);
        {
            nb::gil_scoped_release release;
            hsmm::emission_log_B(g, Y.data(), T, out.data());
        }
        return wrap(std::move(out), {T, std::size_t(g.N)});
    }, "Y"_a, "means"_a, "vars"_a, "df"_a = nb::none());

    m.def("sequence_loglik", [](Arr2 log_B, Arr1 log_pi, Arr2 log_A, Arr2 log_P, bool censored) {
        auto model = make_model(log_pi, log_A, log_P);
        check_log_B(model, log_B);
        nb::gil_scoped_release release;
        return hsmm::sequence_loglik(model, log_B.data(), log_B.shape(0), censored);
    }, "log_B"_a, "log_pi"_a, "log_A"_a, "log_P"_a, "censored"_a = true);

    m.def("viterbi", [](Arr2 log_B, Arr1 log_pi, Arr2 log_A, Arr2 log_P, bool censored, std::size_t max_bytes) {
        auto model = make_model(log_pi, log_A, log_P);
        check_log_B(model, log_B);
        hsmm::ViterbiResult r;
        {
            nb::gil_scoped_release release;
            r = hsmm::viterbi(model, log_B.data(), log_B.shape(0), censored, max_bytes);
        }
        const std::size_t T = r.path.size(), S = r.seg_states.size();
        return nb::make_tuple(wrap(std::move(r.path), {T}), wrap(std::move(r.seg_states), {S}),
                              wrap(std::move(r.seg_durations), {S}), r.final_remaining, r.score);
    }, "log_B"_a, "log_pi"_a, "log_A"_a, "log_P"_a, "censored"_a = true, "max_bytes"_a = std::size_t(1) << 31);

    m.def("em_step", [](Arr2 Y, Arr1 log_pi, Arr2 log_A, Arr2 log_P, Arr2 means, Arr2 vars,
                        const std::optional<Arr1>& df, double var_floor, double dur_smoothing, bool censored,
                        std::size_t max_bytes, bool fit_df) {
        auto model = make_model(log_pi, log_A, log_P);
        auto g = make_emission(means, vars, df);
        if (g.N != model.N) throw std::invalid_argument("means/vars must have N rows");
        if (Y.shape(1) != std::size_t(g.K) || Y.shape(0) < 1) throw std::invalid_argument("Y must be [T, K]");
        hsmm::EmOptions o;
        o.var_floor = var_floor; o.dur_smoothing = dur_smoothing; o.censored = censored; o.max_bytes = max_bytes;
        o.fit_df = fit_df;
        hsmm::EmResult r;
        {
            nb::gil_scoped_release release;
            r = hsmm::em_step(model, g, Y.data(), Y.shape(0), o);
        }
        const std::size_t N = model.N, D = model.D, K = g.K;
        return nb::make_tuple(wrap(std::move(r.log_pi), {N}), wrap(std::move(r.log_A), {N, N}),
                              wrap(std::move(r.log_P), {N, D}), wrap(std::move(r.means), {N, K}),
                              wrap(std::move(r.vars), {N, K}),
                              r.df.empty() ? nb::object(nb::none()) : nb::object(nb::cast(wrap(std::move(r.df), {N}))),
                              r.loglik);
    }, "Y"_a, "log_pi"_a, "log_A"_a, "log_P"_a, "means"_a, "vars"_a, "df"_a = nb::none(), "var_floor"_a = 1e-6,
       "dur_smoothing"_a = 0.0, "censored"_a = true, "max_bytes"_a = std::size_t(1) << 31, "fit_df"_a = true);

    nb::class_<hsmm::Filter>(m, "Filter")
        .def("__init__", [](hsmm::Filter* self, Arr1 log_pi, Arr2 log_A, Arr2 log_P) {
            new (self) hsmm::Filter(make_model(log_pi, log_A, log_P));
        }, "log_pi"_a, "log_A"_a, "log_P"_a)
        .def("reset", &hsmm::Filter::reset)
        .def("step", [](hsmm::Filter& f, Arr1 log_b) {
            const int N = f.model().N;
            if (log_b.shape(0) != std::size_t(N)) throw std::invalid_argument("log_b must have length N");
            std::vector<double> marg(N);
            const double ll = f.step(log_b.data(), marg.data());
            return nb::make_tuple(wrap(std::move(marg), {std::size_t(N)}), ll);
        }, "log_b"_a)
        .def("belief", [](const hsmm::Filter& f) {
            return wrap(std::vector<double>(f.alpha()), {std::size_t(f.model().N), std::size_t(f.model().D)});
        });
}
