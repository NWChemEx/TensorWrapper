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

namespace tensorwrapper::buffer::detail_ {

/** @brief Dispatches to the appropriate backend based on the FP type.
 *
 *
 *
 */
class AdditionVisitor {
public:
    // AdditionVisitor(shape, permutation, shape, permutation)
    template<typename LHSType, typename RHSType>
    void operator()(std::span<LHSType> lhs, std::span<const RHSType> rhs) {
        // auto lhs_wrapped = backends::eigen::wrap_span(lhs);
        // auto rhs_wrapped = backends::eigen::wrap_span(rhs);
        for(std::size_t i = 0; i < lhs.size(); ++i) lhs[i] += rhs[i];
    }
};

} // namespace tensorwrapper::buffer::detail_
