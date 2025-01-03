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
#include <tensorwrapper/shape/shape_fwd.hpp>
#include <utilities/dsl/dsl.hpp>

namespace tensorwrapper::dsl {

/** @brief Object which evaluates the AST of an expression pairwise.
 *
 *  The easiest way to evaluate a tensor network is as a series of assignments,
 *  i.e., things that look like `A = B` and binary operations coupled to
 *  assignments, i.e., things that look like `C = A + B`. That's what this
 *  parser does. It should be noted that this is not necessarily the most
 *  performant way to evaluate the AST, e.g., this prohibits detection of
 *  common intermediates across multiple equations.
 *
 *  @note The
 *        @code
 *        auto pA     = lhs.object().clone();
 *        auto pB     = lhs.object().clone();
 *        auto labels = lhs.labels();
 *        auto lA     = (*pA)(labels);
 *        auto lB     = (*pB)(labels);
 *        dispatch(lA, rhs.lhs());
 *        dispatch(lB, rhs.rhs());
 *        @endcode
 *        are repetitive, but we need to keep pA and pB alive which inhibits
 *        factorization.
 */
class PairwiseParser {
public:
    /** @brief Recursion end-point
     *
     *  Ternary operations like `C = A + B` are ultimately evaluated by
     *  assigning `A` and `B` to temporaries and then summing the temporaries.
     *  The assignment to the temporary ensures that if `A` or `B` is itself a
     *  term it gets evaluated down to an object before the addition happens.
     *  The assignment calls this overload of dispatch.
     *
     *  @param[in] lhs The object to assign @p rhs to.
     *  @param[in] rhs The "expression" that needs to be evaluated.
     *
     */
    template<typename LHSType, typename RHSType>
    void dispatch(LHSType&& lhs, const RHSType& rhs) {
        if constexpr(std::is_floating_point_v<std::decay_t<RHSType>>) {
            lhs.object().scalar_multiplication(rhs);
        } else {
            lhs.object().permute_assignment(lhs.labels(), rhs);
        }
    }

    /** @brief Handles adding two expressions together.
     *
     *  @tparam LHSType The type to assign the sum of @p lhs and @p rhs to.
     *  @tparam T The type of the expression on the left side of the "+" sign.
     *  @tparam U The type of the expression on the right side of the "+" sign.
     *
     *  @param[in] lhs The object that @p rhs will ultimately be assigned to.
     *  @param[in] rhs The expression to evaluate.
     *
     *  @throw std::runtime_error if there is a problem doing the operation.
     *                            Strong throw guarantee.
     */
    template<typename LHSType, typename T, typename U>
    void dispatch(LHSType&& lhs, const utilities::dsl::Add<T, U>& rhs) {
        auto pA     = lhs.object().clone();
        auto pB     = lhs.object().clone();
        auto labels = lhs.labels();
        auto lA     = (*pA)(labels);
        auto lB     = (*pB)(labels);
        dispatch(lA, rhs.lhs());
        dispatch(lB, rhs.rhs());
        lhs.object().addition_assignment(labels, lA, lB);
    }

    /** @brief Handles subtracting two expressions together.
     *
     *  @tparam LHSType The type of the object the expression will be evaluated
     *                  into.
     *  @tparam T The type of the expression on the left side of the "-" sign.
     *  @tparam U The type of the expression on the right side of the "-" sign.
     *
     *  @param[in] lhs The object that @p rhs will ultimately be assigned to.
     *  @param[in] rhs The expression to evaluate.
     *
     *  @throw std::runtime_error if there is a problem doing the operation.
     *                            Strong throw guarantee.
     */
    template<typename LHSType, typename T, typename U>
    void dispatch(LHSType&& lhs, const utilities::dsl::Subtract<T, U>& rhs) {
        auto pA     = lhs.object().clone();
        auto pB     = lhs.object().clone();
        auto labels = lhs.labels();
        auto lA     = (*pA)(labels);
        auto lB     = (*pB)(labels);
        dispatch(lA, rhs.lhs());
        dispatch(lB, rhs.rhs());
        lhs.object().subtraction_assignment(labels, lA, lB);
    }

    /** @brief Handles multiplying two expressions together.
     *
     *  @tparam LHSType The type of the object the expression will be evaluated
     *                  into.
     *  @tparam T The type of the expression on the left side of the "*" sign.
     *  @tparam U The type of the expression on the right side of the "*" sign.
     *
     *  @param[in] lhs The object that @p rhs will ultimately be assigned to.
     *  @param[in] rhs The expression to evaluate.
     *
     *  @throw std::runtime_error if there is a problem doing the operation.
     *                            Strong throw guarantee.
     */
    template<typename LHSType, typename T, typename U>
    void dispatch(LHSType&& lhs, const utilities::dsl::Multiply<T, U>& rhs) {
        auto pA     = lhs.object().clone();
        auto pB     = lhs.object().clone();
        auto labels = lhs.labels();
        auto lA     = (*pA)(labels);
        auto lB     = (*pB)(labels);
        dispatch(lA, rhs.lhs());
        dispatch(lB, rhs.rhs());
        lhs.object().multiplication_assignment(labels, lA, lB);
    }
};

} // namespace tensorwrapper::dsl