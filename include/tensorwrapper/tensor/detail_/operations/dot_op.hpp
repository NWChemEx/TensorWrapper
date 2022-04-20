#pragma once
#include "tensorwrapper/tensor/detail_/op_layer.hpp"
#include "tensorwrapper/tensor/type_traits/type_traits.hpp"

namespace tensorwrapper::tensor::detail_ {

/** @brief Calculates the dot product of two tensors.
 *
 *  @note TA doesn't support dot product between ToT and non-ToT tensors,
 *        so both expressions have to be the same type at the moment.
 *
 *  @tparam T the expression type of the inputs.
 *  @tparam <anonymous> Used to disable this overload if T is not part of
 *                      the expression layer.
 *
 *  @param[in] lhs The expression on the left side of the dot.
 *  @param[in] rhs The expression on the right side of the dot.
 *
 *  @return The value of the dot product.
 */
template<typename T, typename = enable_if_expression_t<std::decay_t<T>>>
double dot(T&& lhs, T&& rhs) {
    auto lhs_variant = lhs.variant(lhs);
    auto rhs_variant = rhs.variant(rhs);

    double rv;
    auto l = [&](auto&& lhs) {
        auto m = [&](auto&& rhs) {
            rv = lhs.dot(rhs);
        };
        std::visit(m, rhs_variant);
    };
    std::visit(l, lhs_variant);
    return rv;
}

} // namespace tensorwrapper::tensor::detail_
