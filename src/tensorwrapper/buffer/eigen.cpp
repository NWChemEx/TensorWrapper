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
#include "detail_/eigen_tensor.hpp"
#include <sstream>
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

#define TPARAMS template<typename FloatType>
#define EIGEN Eigen<FloatType>

// -- Public Methods

TPARAMS
EIGEN::Eigen() noexcept = default;

TPARAMS
EIGEN::Eigen(pimpl_pointer pimpl, layout_pointer playout,
             allocator_base_pointer pallocator) :
  my_base_type(std::move(playout), std::move(pallocator)),
  m_pimpl_(std::move(pimpl)) {}

TPARAMS
EIGEN::Eigen(const Eigen& other) :
  Eigen(other.has_pimpl_() ? other.m_pimpl_->clone() : nullptr, other.layout(),
        other.allocator()) {}

TPARAMS
EIGEN::Eigen(Eigen&& other) noexcept = default;

TPARAMS
EIGEN& EIGEN::operator=(const Eigen& rhs) {
    if(this != &rhs) Eigen(rhs).swap(*this);
    return *this;
}

TPARAMS
EIGEN& EIGEN::operator=(Eigen&& rhs) noexcept = default;

TPARAMS
EIGEN::~Eigen() noexcept = default;

TPARAMS
void EIGEN::swap(Eigen& other) noexcept { m_pimpl_.swap(other.m_pimpl_); }

TPARAMS
bool EIGEN::operator==(const Eigen& rhs) const noexcept {
    if(has_pimpl_() != rhs.has_pimpl_()) return false;
    if(!has_pimpl_()) return true;
    return m_pimpl_->are_equal(*rhs.m_pimpl_);
}

// -- Protected Methods

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::clone_() const {
    return std::make_unique<my_type>(*this);
}

TPARAMS
bool EIGEN::are_equal_(const_buffer_base_reference rhs) const noexcept {
    return my_base_type::template are_equal_impl_<my_type>(rhs);
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::addition_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::addition_assignment_(this_labels, lhs, rhs);
    using alloc_type     = allocator::Eigen<FloatType>;
    const auto& lhs_down = alloc_type::rebind(lhs.object());
    const auto& rhs_down = alloc_type::rebind(rhs.object());
    if(!has_pimpl_()) m_pimpl_ = lhs_down.pimpl_().clone();
    pimpl_().addition_assignment(this_labels, lhs.labels(), rhs.labels(),
                                 lhs_down.pimpl_(), rhs_down.pimpl_());

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::subtraction_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::subtraction_assignment_(this_labels, lhs, rhs);
    using alloc_type     = allocator::Eigen<FloatType>;
    const auto& lhs_down = alloc_type::rebind(lhs.object());
    const auto& rhs_down = alloc_type::rebind(rhs.object());
    if(!has_pimpl_()) m_pimpl_ = lhs_down.pimpl_().clone();
    pimpl_().subtraction_assignment(this_labels, lhs.labels(), rhs.labels(),
                                    lhs_down.pimpl_(), rhs_down.pimpl_());
    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::multiplication_assignment_(this_labels, lhs, rhs);

    using alloc_type     = allocator::Eigen<FloatType>;
    const auto& lhs_down = alloc_type::rebind(lhs.object());
    const auto& rhs_down = alloc_type::rebind(rhs.object());

    if(!has_pimpl_()) m_pimpl_ = lhs_down.pimpl_().clone();
    if(this_labels.is_hadamard_product(lhs.labels(), rhs.labels()))
        pimpl_().hadamard_assignment(this_labels, lhs.labels(), rhs.labels(),
                                     lhs_down.pimpl_(), rhs_down.pimpl_());
    else if(this_labels.is_contraction(lhs.labels(), rhs.labels()))
        pimpl_().contraction_assignment(this_labels, lhs.labels(), rhs.labels(),
                                        this->layout().shape(),
                                        lhs_down.pimpl_(), rhs_down.pimpl_());
    else
        throw std::runtime_error("Mixed products NYI");

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::permute_assignment_(
  label_type this_labels, const_labeled_reference rhs) {
    BufferBase::permute_assignment_(this_labels, rhs);
    using alloc_type     = allocator::Eigen<FloatType>;
    const auto& rhs_down = alloc_type::rebind(rhs.object());
    if(!has_pimpl_()) m_pimpl_ = rhs_down.pimpl_().clone();
    pimpl_().permute_assignment(this_labels, rhs.labels(), rhs_down.pimpl_());

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::scalar_multiplication_(
  label_type this_labels, double scalar, const_labeled_reference rhs) {
    BufferBase::permute_assignment_(this_labels, rhs);
    using alloc_type     = allocator::Eigen<FloatType>;
    const auto& rhs_down = alloc_type::rebind(rhs.object());
    if(!has_pimpl_()) m_pimpl_ = rhs_down.pimpl_().clone();
    pimpl_().scalar_multiplication(this_labels, rhs.labels(), scalar,
                                   rhs_down.pimpl_());
    return *this;
}

TPARAMS
typename EIGEN::pointer EIGEN::data_() noexcept {
    return m_pimpl_ ? m_pimpl_->data() : nullptr;
}

TPARAMS
typename EIGEN::const_pointer EIGEN::data_() const noexcept {
    return m_pimpl_ ? m_pimpl_->data() : nullptr;
}

TPARAMS
typename EIGEN::reference EIGEN::get_elem_(index_vector index) {
    return pimpl_().get_elem(std::move(index));
}

TPARAMS
typename EIGEN::const_reference EIGEN::get_elem_(index_vector index) const {
    return pimpl_().get_elem(std::move(index));
}
TPARAMS
typename EIGEN::polymorphic_base::string_type EIGEN::to_string_() const {
    return m_pimpl_ ? m_pimpl_->to_string() : "";
}

// -- Private methods

TPARAMS
bool EIGEN::has_pimpl_() const noexcept { return static_cast<bool>(m_pimpl_); }

TPARAMS
void EIGEN::assert_pimpl_() const {
    if(has_pimpl_()) return;
    throw std::runtime_error("buffer::Eigen has no PIMPL!");
}

TPARAMS
typename EIGEN::pimpl_reference EIGEN::pimpl_() {
    assert_pimpl_();
    return *m_pimpl_;
}

TPARAMS
typename EIGEN::const_pimpl_reference EIGEN::pimpl_() const {
    assert_pimpl_();
    return *m_pimpl_;
}

#undef EIGEN
#undef TPARAMS

#define DEFINE_EIGEN_BUFFER(TYPE) template class Eigen<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_BUFFER);

#undef DEFINE_EIGEN_BUFFER

} // namespace tensorwrapper::buffer