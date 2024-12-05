#pragma once
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor {

/** @brief Converts tensors to Eigen::tensor then executes the operation.
 *
 *
 */
class Eigen {
public:
    using labeled_tensor = typename Tensor::labeled_tensor;

    static labeled_tensor assign(labeled_tensor lhs, labeled_tensor rhs) {
        if(lhs.labels() != rhs.labels()) { // Transpose needed
        }
    }

    static labeled_tensor add(labeled_tensor result, labeled_tensor lhs,
                              labeled_tensor rhs) {}
};

} // namespace tensorwrapper::dsl::executor