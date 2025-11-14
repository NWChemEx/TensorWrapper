/*
 * Copyright 2025 NWChemEx-Project
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
#include <span>
#include <vector>

namespace tensorwrapper::testing {

template<typename TestType, typename Fxn1, typename Fxn2>
void scalar_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    std::vector<value_type> result_data(1, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(1, value_type{3});
    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({}));
    TestType s0(s0_span, shape_type({}));

    label_type out("");
    label_type rhs("");
    the_op(out, rhs, result, s0);
    REQUIRE(result.get_elem({}) == corr_op(s0_data[0]));
}

} // namespace tensorwrapper::testing
