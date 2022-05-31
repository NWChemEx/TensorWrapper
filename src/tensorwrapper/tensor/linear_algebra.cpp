#include "../ta_helpers/pow.hpp"
#include "../ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/linear_algebra.hpp"
#include "tensorwrapper/tensor/tensor.hpp"
#include <TiledArray/math/linalg/cholesky.h>
#include <TiledArray/math/linalg/svd.h>

namespace tensorwrapper::tensor {

using TWrapper       = ScalarTensorWrapper;
using ta_tensor_type = TA::TSpArrayD;

std::pair<TWrapper, TWrapper> eigen_solve(const TWrapper& X) {
    using tensor_type = ta_tensor_type;

    const auto& x = X.get<tensor_type>();

    auto [eval_vec, evecs] = TA::math::linalg::heig(x);
    const auto& tr1        = evecs.trange().dim(0);
    auto evals = ta_helpers::array_from_vec(eval_vec, tr1, evecs.world());
    TWrapper EVals(std::move(evals));
    TWrapper EVecs(std::move(evecs));
    return std::make_pair(EVals, EVecs);
}

std::pair<TWrapper, TWrapper> eigen_solve(const TWrapper& X,
                                          const TWrapper& S) {
    using tensor_type = ta_tensor_type;

    const auto& x = X.get<tensor_type>();
    const auto& s = S.get<tensor_type>();

    auto [eval_vec, evecs] = TA::math::linalg::heig(x, s);
    const auto& tr1        = evecs.trange().dim(0);
    auto evals = ta_helpers::array_from_vec(eval_vec, tr1, evecs.world());
    TWrapper EVals(std::move(evals));
    TWrapper EVecs(std::move(evecs));
    return std::make_pair(EVals, EVecs);
}

TWrapper cholesky_linv(const TWrapper& M) {
    using tensor_type = ta_tensor_type;

    const auto& m = M.get<tensor_type>();

    auto linv = TA::math::linalg::cholesky_linv(m);
    TWrapper Linv(std::move(linv));

    return Linv;
}

TWrapper hmatrix_pow(const TWrapper& S, double pow) {
    const auto s = S.get<ta_tensor_type>();
    auto s_out   = tensorwrapper::ta_helpers::hmatrix_pow(s, pow);
    return TWrapper(s_out);
}

template<TA::SVD::Vectors Vecs>
auto SVD_(const TWrapper& M) {
    // Flow control booleans
    constexpr bool no_vecs   = (Vecs == TA::SVD::ValuesOnly);
    constexpr bool both_vecs = (Vecs == TA::SVD::AllVectors);

    // Grab the matrix dimension ranges and determine the shorter one
    const auto& m    = M.get<ta_tensor_type>();
    const auto& tr_m = m.trange().dim(0);
    const auto& tr_n = m.trange().dim(1);
    const auto& tr_k = tr_m.extent() < tr_n.extent() ? tr_m : tr_n;

    // Make vector TiledRanges
    TA::TiledRange u_trange{tr_m, tr_k}, vt_trange{tr_k, tr_n};

    // TA does SVD
    auto svd_results = TA::math::linalg::svd<Vecs>(m, u_trange, vt_trange);

    if constexpr(no_vecs) {
        // No vectors, so the result is just the value vector
        // Convert to an array, wrap, and return
        auto s = ta_helpers::array_from_vec(svd_results, tr_k, m.world());
        return TWrapper(std::move(s));
    } else {
        // Grab values and first vector
        auto s_vec = std::get<0>(svd_results);
        auto v1    = std::get<1>(svd_results);

        // Convert value vector to 1D array
        auto s = ta_helpers::array_from_vec(s_vec, tr_k, m.world());

        // Wrap values and vector
        TWrapper S(std::move(s));
        TWrapper V1(std::move(v1));

        // Determine if there is another vector, then return
        if constexpr(both_vecs) {
            auto v2 = std::get<2>(svd_results);
            TWrapper V2(std::move(v2));
            return std::make_tuple(S, V1, V2);
        } else {
            return std::make_pair(S, V1);
        }
    }
}

TWrapper SVDValues(const TWrapper& M) { return SVD_<TA::SVD::ValuesOnly>(M); }

std::pair<TWrapper, TWrapper> SVDLeft(const TWrapper& M) {
    return SVD_<TA::SVD::LeftVectors>(M);
}

std::pair<TWrapper, TWrapper> SVDRight(const TWrapper& M) {
    return SVD_<TA::SVD::RightVectors>(M);
}

std::tuple<TWrapper, TWrapper, TWrapper> SVD(const TWrapper& M) {
    return SVD_<TA::SVD::AllVectors>(M);
}

} // namespace tensorwrapper::tensor
