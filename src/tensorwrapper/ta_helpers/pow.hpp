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

#pragma once
#include <TiledArray/math/linalg/heig.h>
#include <TiledArray/special/diagonal_array.h>
#include <algorithm>
#include <cmath>

namespace tensorwrapper::ta_helpers {

/** @brief Raises a Hermitian matrix to an arbitrary power.
 *
 *  This function will raise a Hermitian matrix to a power. The power need not
 *  be an integer nor positive (*e.g.*, it can be -0.5). The algorithm used in
 *  this function assumes a Hermitian matrix and will produce erroneous results
 *  if @p t is not Hermitian.
 *
 *  @tparam TensorType The type of the tensor. Assumed to be a TA::DistArray
 *                     instance.
 *  @param[in] t The tensor being raised to the power. The tensor is assumed to
 *               be Hermitian.
 *  @param[in] p The power the tensor is being raised to.
 *  @return The result of raising @p t to the power @p p.
 */
template<typename TensorType>
auto hmatrix_pow(const TensorType& t, double p) {
    const auto& tr = t.trange();
    auto& world    = t.world();

    TA_ASSERT(tr.rank() == 2);
    TA_ASSERT(tr.dim(0) == tr.dim(1));

    // Step 1: Diagonalize the matrix
    auto [evals, evecs] = TA::heig(t);

    // Step 2: Raise eigenvalues to power p
    using std::pow;
    std::for_each(evals.begin(), evals.end(), [=](auto& x) { x = pow(x, p); });

    // Step 3: Turn eigenvalues into a diagonal matrix
    auto evals_tensor =
      TA::diagonal_array<TensorType>(world, tr, evals.begin(), evals.end());

    // Step 4: Rotate t**p back to the original basis set
    TensorType rv;
    rv("i,j") = evecs("i,a") * evals_tensor("a,b") * evecs("j,b");
    return rv;
}

} // namespace tensorwrapper::ta_helpers
