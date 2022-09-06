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

#include "../ta_helpers/ta_helpers.hpp"
#include "conversion/conversion.hpp"
#include "detail_/ta_to_tw.hpp"
#include <tensorwrapper/tensor/apply_elementwise.hpp>

namespace tensorwrapper::tensor {

using tensor = ScalarTensorWrapper;

tensor apply_elementwise(const tensor& input,
                         const std::function<double(double)>& fxn) {
    to_ta_distarrayd_t converter;
    const auto& t = converter.convert(input.buffer());
    return detail_::ta_to_tw(
      tensorwrapper::ta_helpers::apply_elementwise(t, fxn));
}

void apply_elementwise_inplace(tensor& input,
                               const std::function<void(double&)>& fxn) {
    to_ta_distarrayd_t converter;
    auto& t = converter.convert(input.buffer());
    tensorwrapper::ta_helpers::apply_elementwise_inplace(t, fxn);
}

} // namespace tensorwrapper::tensor
