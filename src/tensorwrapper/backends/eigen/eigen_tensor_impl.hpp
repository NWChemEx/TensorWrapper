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
#include "eigen_tensor.hpp"
#include <iomanip>
#include <span>
#include <sstream>
#include <tensorwrapper/detail_/integer_utilities.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

namespace tensorwrapper::backends::eigen {

/// Implements EigenTensor by wrapping eigen::TensorMap
template<typename FloatType, unsigned int Rank>
class EigenTensorImpl : public EigenTensor<FloatType> {
private:
    /// Type of *this
    using my_type = EigenTensorImpl<FloatType, Rank>;

    /// Type *this inherits from
    using base_type = EigenTensor<FloatType>;

public:
    using eigen_tensor_type = Eigen::Tensor<FloatType, Rank, Eigen::RowMajor>;
    using eigen_data_type   = Eigen::TensorMap<eigen_tensor_type>;
    using eigen_reference   = eigen_data_type&;
    using const_eigen_reference = const eigen_data_type&;

    ///@{
    using typename base_type::const_reference;
    using typename base_type::const_shape_reference;
    using typename base_type::eigen_rank_type;
    using typename base_type::index_vector;
    using typename base_type::label_type;
    using typename base_type::reference;
    using typename base_type::size_type;
    using typename base_type::string_type;
    using typename base_type::value_type;
    ///@}

    EigenTensorImpl(std::span<value_type> data, const_shape_reference shape) :
      m_tensor_(
        make_from_shape_(data, shape, std::make_index_sequence<Rank>())) {}

    EigenTensorImpl permute(label_type perm) const;

protected:
    /// Implement rank by returning template parameter
    eigen_rank_type rank_() const noexcept override { return Rank; }

    /// Calls Eigen's size() method to implement size()
    size_type size_() const noexcept override { return m_tensor_.size(); }

    /// Calls Eigen's dimension(i) method to implement extent(i)
    size_type extent_(eigen_rank_type i) const override {
        return m_tensor_.dimension(i);
    }

    /// Unwraps index vector into Eigen's operator() to get element
    const_reference get_elem_(index_vector index) const override;

    /// Unwraps index vector into Eigen's operator() to set element
    void set_elem_(index_vector index, value_type new_value) override;

    /// Calls std::fill to set the values
    void fill_(value_type value) override;

    /// Calls add_to_stream_ on a stringstream to implement to_string
    string_type to_string_() const override;

    /// Relies on Eigen's operator<< to add to stream
    std::ostream& add_to_stream_(std::ostream& os) const override;

    void addition_assignment_(label_type this_label, label_type lhs_label,
                              label_type rhs_label, const base_type& lhs,
                              const base_type& rhs) override;

    void subtraction_assignment_(label_type this_label, label_type lhs_label,
                                 label_type rhs_label, const base_type& lhs,
                                 const base_type& rhs) override;

    void hadamard_assignment_(label_type this_label, label_type lhs_label,
                              label_type rhs_label, const base_type& lhs,
                              const base_type& rhs) override;

    void permute_assignment_(label_type this_label, label_type rhs_label,
                             const base_type& rhs) override;

    void scalar_multiplication_(label_type this_label, label_type rhs_label,
                                FloatType scalar,
                                const base_type& rhs) override;

    void contraction_assignment_(label_type this_labels, label_type lhs_labels,
                                 label_type rhs_labels, const base_type& lhs,
                                 const base_type& rhs) override;

private:
    // Code factorization for implementing element-wise operations
    template<typename OperationType>
    void element_wise_op_(OperationType op, label_type this_label,
                          label_type lhs_label, label_type rhs_label,
                          const base_type& lhs, const base_type& rhs);

    // Handles TMP needed to create an Eigen TensorMap from a Smooth object
    template<std::size_t... I>
    auto make_from_shape_(std::span<value_type> data,
                          const_shape_reference shape,
                          std::index_sequence<I...>) {
        return eigen_data_type(data.data(), shape.extent(I)...);
    }

    // Gets an element from the Eigen Tensor by unwrapping a std::vector
    template<std::size_t... I>
    reference unwrap_vector_(index_vector index, std::index_sequence<I...>) {
        return m_tensor_(tensorwrapper::detail_::to_long(index.at(I))...);
    }

    // Same as mutable version, but result is read-only
    template<std::size_t... I>
    const_reference unwrap_vector_(index_vector index,
                                   std::index_sequence<I...>) const {
        return m_tensor_(tensorwrapper::detail_::to_long(index.at(I))...);
    }

    // The Eigen tensor *this wraps
    eigen_data_type m_tensor_;
};

template<typename FloatType>
std::unique_ptr<EigenTensor<FloatType>> make_eigen_tensor(
  std::span<FloatType> data, shape::SmoothView<const shape::Smooth> shape);

#define DECLARE_EIGEN_TENSOR(TYPE)                  \
    extern template class EigenTensorImpl<TYPE, 0>; \
    extern template class EigenTensorImpl<TYPE, 1>; \
    extern template class EigenTensorImpl<TYPE, 2>; \
    extern template class EigenTensorImpl<TYPE, 3>; \
    extern template class EigenTensorImpl<TYPE, 4>; \
    extern template class EigenTensorImpl<TYPE, 5>; \
    extern template class EigenTensorImpl<TYPE, 6>; \
    extern template class EigenTensorImpl<TYPE, 7>; \
    extern template class EigenTensorImpl<TYPE, 8>; \
    extern template class EigenTensorImpl<TYPE, 9>; \
    extern template class EigenTensorImpl<TYPE, 10>

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_EIGEN_TENSOR);

#undef DECLARE_EIGEN_TENSOR

} // namespace tensorwrapper::backends::eigen
