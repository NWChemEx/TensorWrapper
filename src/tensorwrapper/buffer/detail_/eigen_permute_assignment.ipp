#pragma once
#include "eigen_tensor.hpp"

namespace tensorwrapper::buffer::detail_ {

template<typename FloatType, unsigned int Rank>
void EigenTensor<FloatType, Rank>::permute_assignment_(
  label_type this_labels, label_type rhs_labels, const_pimpl_reference rhs) {
    using my_type        = EigenTensor<FloatType, Rank>;
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if(this_labels != rhs_labels) { // We need to permute rhs before assignment
        // Eigen adopts the opposite definition of permutation from us.
        auto r_to_l = this_labels.permutation(rhs_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_down->value().shuffle(r_to_l2);
    } else {
        m_tensor_ = rhs_down->value();
    }
}

} // namespace tensorwrapper::buffer::detail_