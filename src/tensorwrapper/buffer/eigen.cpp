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
typename EIGEN::pointer EIGEN::get_mutable_data_() noexcept {
    return m_pimpl_ ? m_pimpl_->get_mutable_data() : nullptr;
}

TPARAMS
typename EIGEN::const_pointer EIGEN::get_immutable_data_() const noexcept {
    return m_pimpl_ ? m_pimpl_->get_immutable_data() : nullptr;
}

TPARAMS
typename EIGEN::const_reference EIGEN::get_elem_(index_vector index) const {
    return pimpl_().get_elem(std::move(index));
}

TPARAMS
void EIGEN::set_elem_(index_vector index, element_type new_value) {
    return pimpl_().set_elem(std::move(index), std::move(new_value));
}

TPARAMS
typename EIGEN::const_reference EIGEN::get_data_(size_type index) const {
    return pimpl_().get_data(std::move(index));
}

TPARAMS
void EIGEN::set_data_(size_type index, element_type new_value) {
    return pimpl_().set_data(std::move(index), std::move(new_value));
}

TPARAMS
void EIGEN::fill_(element_type value) {
    return pimpl_().fill(std::move(value));
}

TPARAMS
void EIGEN::copy_(const element_vector& values) {
    return pimpl_().copy(values);
}

TPARAMS
typename EIGEN::polymorphic_base::string_type EIGEN::to_string_() const {
    return m_pimpl_ ? m_pimpl_->to_string() : "";
}

TPARAMS
std::ostream& EIGEN::add_to_stream_(std::ostream& os) const {
    return m_pimpl_ ? m_pimpl_->add_to_stream(os) : os;
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

TPARAMS
EIGEN& to_eigen_buffer(BufferBase& b) {
    using allocator_type = allocator::Eigen<FloatType>;
    return allocator_type::rebind(b);
}

TPARAMS
const EIGEN& to_eigen_buffer(const BufferBase& b) {
    using allocator_type = allocator::Eigen<FloatType>;
    return allocator_type::rebind(b);
}

#undef EIGEN
#undef TPARAMS

#define DEFINE_EIGEN_BUFFER(TYPE) template class Eigen<TYPE>
#define DEFINE_TO_EIGEN_BUFFER(TYPE) \
    template Eigen<TYPE>& to_eigen_buffer(BufferBase&)
#define DEFINE_TO_CONST_EIGEN_BUFFER(TYPE) \
    template const Eigen<TYPE>& to_eigen_buffer(const BufferBase&)

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_BUFFER);
TW_APPLY_FLOATING_POINT_TYPES(DEFINE_TO_EIGEN_BUFFER);
TW_APPLY_FLOATING_POINT_TYPES(DEFINE_TO_CONST_EIGEN_BUFFER);

#undef DEFINE_EIGEN_BUFFER
#undef DEFINE_TO_EIGEN_BUFFER
#undef DEFINE_TO_CONST_EIGEN_BUFFER

} // namespace tensorwrapper::buffer