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

#include "../ta_helpers/remove_redundancy.hpp"
#include "conversion/conversion.hpp"
#include "detail_/ta_to_tw.hpp"
#include <tensorwrapper/tensor/remove_redundancy.hpp>

namespace tensorwrapper::tensor {

ScalarTensorWrapper remove_redundancy(const ScalarTensorWrapper& C,
                                      const ScalarTensorWrapper& S,
                                      double thresh) {
    to_ta_distarrayd_t converter;
    const auto& C_ta = converter.convert(C.buffer());
    const auto& S_ta = converter.convert(S.buffer());
    auto new_C       = ta_helpers::remove_redundancy(C_ta, S_ta, thresh);
    return detail_::ta_to_tw(std::move(new_C));
}

} // namespace tensorwrapper::tensor
