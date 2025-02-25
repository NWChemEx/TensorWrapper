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
#include "../../backends/eigen.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>

namespace tensorwrapper::buffer::detail_ {

template<typename FloatType>
class EigenPIMPL
  : public tensorwrapper::detail_::PolymorphicBase<EigenPIMPL<FloatType>> {
private:
    using my_type          = EigenPIMPL<FloatType>;
    using polymorphic_base = tensorwrapper::detail_::PolymorphicBase<my_type>;

public:
    using parent_type           = Eigen<FloatType>;
    using pimpl_pointer         = typename parent_type::pimpl_pointer;
    using label_type            = typename parent_type::label_type;
    using reference             = typename parent_type::reference;
    using const_shape_reference = const shape::ShapeBase&;
    using const_reference       = typename parent_type::const_reference;
    using pointer               = typename parent_type::pointer;
    using const_pointer         = typename parent_type::const_pointer;
    using string_type           = typename parent_type::string_type;
    using index_vector          = typename parent_type::index_vector;
    using size_type             = typename parent_type::size_type;

    using const_pimpl_reference = const EigenPIMPL<FloatType>&;

    using eigen_rank_type = unsigned int;

    eigen_rank_type rank() const noexcept { return rank_(); }

    size_type extent(eigen_rank_type i) const { return extent_(i); }

    pointer data() noexcept { return data_(); }

    const_pointer data() const noexcept { return data_(); }

    reference get_elem(index_vector index) {
        return get_elem_(std::move(index));
    }

    const_reference get_elem(index_vector index) const {
        return get_elem_(std::move(index));
    }

    void addition_assignment(label_type this_labels, label_type lhs_labels,
                             label_type rhs_labels, const_pimpl_reference lhs,
                             const_pimpl_reference rhs) {
        addition_assignment_(std::move(this_labels), std::move(lhs_labels),
                             std::move(rhs_labels), lhs, rhs);
    }

    void subtraction_assignment(label_type this_labels, label_type lhs_labels,
                                label_type rhs_labels,
                                const_pimpl_reference lhs,
                                const_pimpl_reference rhs) {
        subtraction_assignment_(std::move(this_labels), std::move(lhs_labels),
                                std::move(rhs_labels), lhs, rhs);
    }

    void hadamard_assignment(label_type this_labels, label_type lhs_labels,
                             label_type rhs_labels, const_pimpl_reference lhs,
                             const_pimpl_reference rhs) {
        hadamard_assignment_(std::move(this_labels), std::move(lhs_labels),
                             std::move(rhs_labels), lhs, rhs);
    }

    void contraction_assignment(label_type this_labels, label_type lhs_labels,
                                label_type rhs_labels,
                                const_shape_reference result_shape,
                                const_pimpl_reference lhs,
                                const_pimpl_reference rhs) {
        contraction_assignment_(std::move(this_labels), std::move(lhs_labels),
                                std::move(rhs_labels), result_shape, lhs, rhs);
    }

    void permute_assignment(label_type this_labels, label_type rhs_labels,
                            const_pimpl_reference rhs) {
        permute_assignment_(std::move(this_labels), std::move(rhs_labels), rhs);
    }

    void scalar_multiplication(label_type this_labels, label_type rhs_labels,
                               FloatType scalar, const_pimpl_reference rhs) {
        scalar_multiplication_(std::move(this_labels), std::move(rhs_labels),
                               scalar, rhs);
    }

protected:
    virtual eigen_rank_type rank_() const noexcept                  = 0;
    virtual size_type extent_(eigen_rank_type i) const              = 0;
    virtual pointer data_() noexcept                                = 0;
    virtual const_pointer data_() const noexcept                    = 0;
    virtual reference get_elem_(index_vector index)                 = 0;
    virtual const_reference get_elem_(index_vector index) const     = 0;
    virtual void addition_assignment_(label_type this_labels,
                                      label_type lhs_labels,
                                      label_type rhs_labels,
                                      const_pimpl_reference lhs,
                                      const_pimpl_reference rhs)    = 0;
    virtual void subtraction_assignment_(label_type this_labels,
                                         label_type lhs_labels,
                                         label_type rhs_labels,
                                         const_pimpl_reference lhs,
                                         const_pimpl_reference rhs) = 0;
    virtual void hadamard_assignment_(label_type this_labels,
                                      label_type lhs_labels,
                                      label_type rhs_labels,
                                      const_pimpl_reference lhs,
                                      const_pimpl_reference rhs)    = 0;
    virtual void contraction_assignment_(label_type this_labels,
                                         label_type lhs_labels,
                                         label_type rhs_labels,
                                         const_shape_reference result_shape,
                                         const_pimpl_reference lhs,
                                         const_pimpl_reference rhs) = 0;
    virtual void permute_assignment_(label_type this_labels,
                                     label_type rhs_labels,
                                     const_pimpl_reference rhs)     = 0;
    virtual void scalar_multiplication_(label_type this_labels,
                                        label_type rhs_labels, FloatType scalar,
                                        const_pimpl_reference rhs)  = 0;
};

} // namespace tensorwrapper::buffer::detail_