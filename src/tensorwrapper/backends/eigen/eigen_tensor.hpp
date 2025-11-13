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
#include <ostream>
#include <string>
#include <tensorwrapper/shape/smooth_view.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>
#include <vector>

namespace tensorwrapper::backends::eigen {

/** @brief API for interacting with Eigen's tensor object.
 *
 *  @tparam FloatType The floating-point type stored in the tensor.
 *
 *  This class defines the API for interacting with Eigen's tensor objects.
 *  Unfortunately, Eigen's tensor objects are templated on the rank of the
 *  tensor (and some other stuff) which makes it hard to deal with them
 *  generically. This class gets the templating down to just the floating-point
 *  type.
 *
 *   N.b. this class wraps Eigen::TensorMap objects, not Eigen::Tensor objects
 *   so as to avoid needing to transfer the data to Eigen. Idea is these classes
 *   can be made on-demand since they just wrap pointers.
 */
template<typename FloatType>
class EigenTensor {
private:
    /// Type of *this
    using my_type = EigenTensor<FloatType>;

public:
    /// Type of an element in *this
    using value_type = FloatType;

    /// Type of a reference to an element in *this
    using reference = value_type&;

    /// Type of a read-only reference to an element in *this
    using const_reference = const value_type&;

    /// Type used to express the shape of *this
    using shape_type = shape::Smooth;

    /// Type of a view acting like a read-only shape_type
    using const_shape_reference = shape::SmoothView<const shape::Smooth>;

    /// Type Eigen uses to express tensor rank
    using eigen_rank_type = unsigned int;

    /// Type used to express sizes and extents
    using size_type = std::size_t;

    /// Type used toe express multi-dimensional indices
    using index_vector = std::vector<size_type>;

    /// Type used to express strings
    using string_type = std::string;

    /// Type used to specify a permutation
    using permutation_type = symmetry::Permutation;

    /// Type of a read-only reference to an object of permutation_type
    using const_permutation_reference = const permutation_type&;

    /** @brief Retrieves the rank of the wrapped tensor.
     *
     *  @return The rank of the wrapped tensor.
     */
    eigen_rank_type rank() const noexcept { return rank_(); }

    size_type size() const noexcept { return size_(); }

    size_type extent(eigen_rank_type i) const {
        assert(i < rank());
        return extent_(i);
    }

    const_reference get_elem(index_vector index) const {
        assert(index.size() == rank());
        return get_elem_(std::move(index));
    }

    void set_elem(index_vector index, value_type new_value) {
        assert(index.size() == rank());
        set_elem_(index, new_value);
    }

    void fill(value_type value) { fill_(std::move(value)); }

    string_type to_string() const { return to_string_(); }

    std::ostream& add_to_stream(std::ostream& os) const {
        return add_to_stream_(os);
    }

    void addition_assignment(const_permutation_reference lhs_permute,
                             const_permutation_reference rhs_permute,
                             const EigenTensor& lhs, const EigenTensor& rhs) {
        return addition_assignment_(lhs_permute, rhs_permute, lhs, rhs);
    }

    void subtraction_assignment(const_permutation_reference lhs_permute,
                                const_permutation_reference rhs_permute,
                                const EigenTensor& lhs,
                                const EigenTensor& rhs) {
        return subtraction_assignment_(lhs_permute, rhs_permute, lhs, rhs);
    }

    void hadamard_assignment(const_permutation_reference lhs_permute,
                             const_permutation_reference rhs_permute,
                             const EigenTensor& lhs, const EigenTensor& rhs) {
        return hadamard_assignment_(lhs_permute, rhs_permute, lhs, rhs);
    }

    void permute_assignment(const_permutation_reference rhs_permute,
                            const EigenTensor& rhs) {
        return permute_assignment_(rhs_permute, rhs);
    }

    void scalar_multiplication(const_permutation_reference rhs_permute,
                               FloatType scalar, const EigenTensor& rhs) {
        return scalar_multiplication_(rhs_permute, scalar, rhs);
    }

    // void contraction_assignment(label_type this_labels, label_type
    // lhs_labels,
    //                             label_type rhs_labels,
    //                             const_shape_reference result_shape,
    //                             const_pimpl_reference lhs,
    //                             const_pimpl_reference rhs) {
    //     contraction_assignment_(std::move(this_labels),
    //     std::move(lhs_labels),
    //                             std::move(rhs_labels), result_shape, lhs,
    //                             rhs);
    // }

protected:
    EigenTensor() noexcept = default;

    virtual eigen_rank_type rank_() const noexcept                   = 0;
    virtual size_type size_() const                                  = 0;
    virtual size_type extent_(eigen_rank_type i) const               = 0;
    virtual const_reference get_elem_(index_vector index) const      = 0;
    virtual void set_elem_(index_vector index, value_type new_value) = 0;
    virtual void fill_(value_type value)                             = 0;
    virtual string_type to_string_() const                           = 0;
    virtual std::ostream& add_to_stream_(std::ostream& os) const     = 0;

    virtual void addition_assignment_(const_permutation_reference lhs_permute,
                                      const_permutation_reference rhs_permute,
                                      const EigenTensor& lhs,
                                      const EigenTensor& rhs) = 0;

    virtual void subtraction_assignment_(
      const_permutation_reference lhs_permute,
      const_permutation_reference rhs_permute, const EigenTensor& lhs,
      const EigenTensor& rhs) = 0;

    virtual void hadamard_assignment_(const_permutation_reference lhs_permute,
                                      const_permutation_reference rhs_permute,
                                      const EigenTensor& lhs,
                                      const EigenTensor& rhs) = 0;

    virtual void permute_assignment_(const_permutation_reference rhs_permute,
                                     const EigenTensor& rhs) = 0;

    virtual void scalar_multiplication_(const_permutation_reference rhs_permute,
                                        FloatType scalar,
                                        const EigenTensor& rhs) = 0;

    // virtual void contraction_assignment_(label_type this_labels,
    //                                      label_type lhs_labels,
    //                                      label_type rhs_labels,
    //                                      const_shape_reference result_shape,
    //                                      const_pimpl_reference lhs,
    //                                      const_pimpl_reference rhs)    = 0;
};

} // namespace tensorwrapper::backends::eigen
