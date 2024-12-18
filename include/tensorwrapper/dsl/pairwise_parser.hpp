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

/** @brief Object which evaluates the AST of an expression pairwise.
 *
 *  @tparam ObjectType The type of the objects associated with the dummy
 *                     indices. Expected to be possibly cv-qualified versions
 *                     of Tensor, buffers, shapes, etc.
 *  @tparam LabelType The type of object used for the dummy indices.
 *
 *  The easiest way to evaluate an abstract syntax tree which contains
 *  operations involving at most two objects is by splitting it into subtrees
 *  which contain at most two connected nodes, i.e., considering each operation
 *  pairwise. That's what this parser does.
 */
template<typename ObjectType, typename LabelType>
class PairwiseParser {
public:
    /// Type of a leaf in the AST
    using labeled_type = Labeled<ObjectType, LabelType>;

    /** @brief Recursion end-point
     *
     *  Evaluates @p rhs given that it will be evaluated into lhs.
     *  This is the natural end-point for recursion down a branch of the AST.
     *
     *  N.b., this overload is only responsible for evaluating @p rhs NOT for
     *  assigning it to @p lhs.
     *
     *  @param[in] lhs The object that @p rhs will ultimately be assigned to.
     *  @param[in] rhs The "expression" that needs to be evaluated.
     *
     *  @return @p rhs untouched.
     *
     *  @throw None No throw guarantee.
     */
    auto dispatch(labeled_type lhs, labeled_type rhs) { return rhs; }

    /** @brief Handles adding two expressions together.
     *
     *  @tparam T The type of the expression on the left side of the "+" sign.
     *  @tparam U The type of the expression on the right side of the "+" sign.
     *
     *  @param[in] lhs The object that @p rhs will ultimately be assigned to.
     *  @param[in] rhs The expression to evaluate.
     *
     *
     */
    template<typename T, typename U>
    auto dispatch(labeled_type lhs, const utilities::dsl::Add<T, U>& rhs) {
        // TODO: This shouldn't be assigning to lhs, but letting the layer up
        // do that
        auto lA = dispatch(lhs, rhs.lhs());
        auto lB = dispatch(lhs, rhs.rhs());
        return add(std::move(lhs), std::move(lA), std::move(lB));
    }

protected:
    labeled_type add(labeled_type result, labeled_type lhs, labeled_type rhs);
};

extern template class PairwiseParser<Tensor, std::string>;

} // namespace dsl
} // namespace tensorwrapper