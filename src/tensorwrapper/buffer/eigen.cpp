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

#include <sstream>
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

using dummy_indices_type = dsl::DummyIndices<std::string>;

#define TPARAMS template<typename FloatType, unsigned short Rank>
#define EIGEN Eigen<FloatType, Rank>

// TPARAMS
// typename EIGEN::buffer_base_reference EIGEN::addition_assignment_(
//   label_type this_labels, const_labeled_buffer_reference rhs) {
//     // TODO layouts
//     if(layout() != rhs.lhs().layout())
//         throw std::runtime_error("Layouts must be the same (for now)");

//     dummy_indices_type llabels(this_labels);
//     dummy_indices_type rlabels(rhs.rhs());

//     using allocator_type       = allocator::Eigen<FloatType, Rank>;
//     const auto& rhs_downcasted = allocator_type::rebind(rhs.lhs());

//     if(llabels != rlabels) {
//         auto r_to_l = rlabels.permutation(llabels);
//         std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
//         m_tensor_ += rhs_downcasted.value().shuffle(r_to_l2);
//     } else {
//         m_tensor_ += rhs_downcasted.value();
//     }

//     return *this;
// }

// TPARAMS
// typename EIGEN::buffer_base_reference EIGEN::permute_assignment_(
//   label_type this_labels, const_labeled_buffer_reference rhs) {
//     dummy_indices_type llabels(this_labels);
//     dummy_indices_type rlabels(rhs.rhs());

//     using allocator_type       = allocator::Eigen<FloatType, Rank>;
//     const auto& rhs_downcasted = allocator_type::rebind(rhs.lhs());

//     if(llabels != rlabels) { // We need to permute rhs before assignment
//         auto r_to_l = rlabels.permutation(llabels);
//         // Eigen wants int objects
//         std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
//         m_tensor_ = rhs_downcasted.value().shuffle(r_to_l2);
//     } else {
//         m_tensor_ = rhs_downcasted.value();
//     }

//     // TODO: permute layout

//     return *this;
// }

TPARAMS
typename EIGEN::string_type EIGEN::to_string_() const {
    std::stringstream ss;
    ss << m_tensor_;
    return ss.str();
}

#undef EIGEN
#undef TPARAMS

#define DEFINE_EIGEN_BUFFER(RANK)      \
    template class Eigen<float, RANK>; \
    template class Eigen<double, RANK>

DEFINE_EIGEN_BUFFER(0);
DEFINE_EIGEN_BUFFER(1);
DEFINE_EIGEN_BUFFER(2);
DEFINE_EIGEN_BUFFER(3);
DEFINE_EIGEN_BUFFER(4);
DEFINE_EIGEN_BUFFER(5);
DEFINE_EIGEN_BUFFER(6);
DEFINE_EIGEN_BUFFER(7);
DEFINE_EIGEN_BUFFER(8);
DEFINE_EIGEN_BUFFER(9);
DEFINE_EIGEN_BUFFER(10);

#undef DEFINE_EIGEN_BUFFER

} // namespace tensorwrapper::buffer