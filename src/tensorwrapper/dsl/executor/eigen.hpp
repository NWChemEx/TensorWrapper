#pragma once
#include "detail_/eigen_dispatcher.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor {

/** @brief Converts tensors to Eigen::tensor then executes the operation.
 *
 *
 */
class Eigen {
public:
    using labeled_tensor = typename tensorwrapper::Tensor::labeled_tensor_type;

    static labeled_tensor assign(labeled_tensor lhs, labeled_tensor rhs) {
        return lhs;
    }

    static labeled_tensor add(labeled_tensor result, labeled_tensor lhs,
                              labeled_tensor rhs) {
        return rhs;
    }
};

} // namespace tensorwrapper::dsl::executor