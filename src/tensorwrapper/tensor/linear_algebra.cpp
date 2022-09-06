/*
 * Copyright 2022 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../ta_helpers/pow.hpp"
#include "../ta_helpers/ta_helpers.hpp"
#include "conversion/conversion.hpp"
#include "detail_/ta_to_tw.hpp"
#include <TiledArray/math/linalg/basic.h>
#include <TiledArray/math/linalg/cholesky.h>
#include <TiledArray/math/linalg/svd.h>
#include <tensorwrapper/tensor/linear_algebra.hpp>
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper::tensor {

using TWrapper = ScalarTensorWrapper;

std::pair<TWrapper, TWrapper> eigen_solve(const TWrapper& X) {
    to_ta_distarrayd_t converter;
    const auto& x = converter.convert(X.buffer());

    auto [eval_vec, evecs] = TA::math::linalg::heig(x);
    const auto& tr1        = evecs.trange().dim(0);
    auto evals = ta_helpers::array_from_vec(eval_vec, tr1, evecs.world());

    auto EVals = detail_::ta_to_tw(std::move(evals));
    auto EVecs = detail_::ta_to_tw(std::move(evecs));
    return std::make_pair(EVals, EVecs);
}

std::pair<TWrapper, TWrapper> eigen_solve(const TWrapper& X,
                                          const TWrapper& S) {
    to_ta_distarrayd_t converter;
    const auto& x = converter.convert(X.buffer());
    const auto& s = converter.convert(S.buffer());

    auto [eval_vec, evecs] = TA::math::linalg::heig(x, s);
    const auto& tr1        = evecs.trange().dim(0);
    auto evals = ta_helpers::array_from_vec(eval_vec, tr1, evecs.world());
    auto EVals = detail_::ta_to_tw(std::move(evals));
    auto EVecs = detail_::ta_to_tw(std::move(evecs));
    return std::make_pair(EVals, EVecs);
}

TWrapper cholesky_linv(const TWrapper& M) {
    to_ta_distarrayd_t converter;
    const auto& m = converter.convert(M.buffer());
    auto linv     = TA::math::linalg::cholesky_linv(m);

    return detail_::ta_to_tw(std::move(linv));
}

TWrapper hmatrix_pow(const TWrapper& S, double pow) {
    to_ta_distarrayd_t converter;
    const auto s = converter.convert(S.buffer());
    auto s_out   = tensorwrapper::ta_helpers::hmatrix_pow(s, pow);

    return detail_::ta_to_tw(s_out);
}

template<TA::SVD::Vectors Vecs>
auto SVD_(const TWrapper& M) {
    // Flow control booleans
    constexpr bool no_vecs   = (Vecs == TA::SVD::ValuesOnly);
    constexpr bool both_vecs = (Vecs == TA::SVD::AllVectors);

    // Grab the matrix dimension ranges and determine the shorter one
    to_ta_distarrayd_t converter;
    const auto& m    = converter.convert(M.buffer());
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
        return detail_::ta_to_tw(std::move(s));
    } else {
        // Grab values and first vector
        auto s_vec = std::get<0>(svd_results);
        auto v1    = std::get<1>(svd_results);

        // Convert value vector to 1D array
        auto s = ta_helpers::array_from_vec(s_vec, tr_k, m.world());

        // Wrap values and vector
        auto S  = detail_::ta_to_tw(std::move(s));
        auto V1 = detail_::ta_to_tw(std::move(v1));

        // Determine if there is another vector, then return
        if constexpr(both_vecs) {
            auto v2 = std::get<2>(svd_results);
            auto V2 = detail_::ta_to_tw(std::move(v2));
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
