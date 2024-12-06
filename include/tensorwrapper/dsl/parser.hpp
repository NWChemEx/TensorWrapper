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