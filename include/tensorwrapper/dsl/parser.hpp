/*
 * Copyright 2024 NWChemEx-Project
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
#include <tensorwrapper/dsl/dsl_forward.hpp>
#include <utilities/dsl/dsl.hpp>

namespace tensorwrapper {
class Tensor;
namespace dsl {

/** @brief Object which walks the AST of an expression.
 *
 */
template<typename ObjectType, typename LabelType>
class Parser {
public:
    /// Type of a leaf in the AST
    using labeled_type = Labeled<ObjectType, LabelType>;

    /** @brief Recursion end-point
     *
     *
     */
    auto dispatch(labeled_type lhs, labeled_type rhs) {
        return assign(std::move(lhs), std::move(rhs));
    }

    template<typename T, typename U>
    auto dispatch(labeled_type lhs, const utilities::dsl::Add<T, U>& rhs) {
        auto lA = dispatch(lhs, rhs.lhs());
        auto lB = dispatch(lhs, rhs.rhs());
        return add(std::move(lhs), std::move(lA), std::move(lB));
    }

protected:
    labeled_type assign(labeled_type lhs, labeled_type rhs);
    labeled_type add(labeled_type result, labeled_type lhs, labeled_type rhs);
};

extern template class Parser<Tensor, std::string>;

} // namespace dsl
} // namespace tensorwrapper