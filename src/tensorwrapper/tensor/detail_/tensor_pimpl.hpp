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

#pragma once
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper::detail_ {

/** @brief Actually implements a Tensor object.
 *
 *  At a high-level Tensors have two things: a logical layout and a buffer.
 *  This class wraps that state together.
 *
 */
class TensorPIMPL {
public:
    /// Type *this implements
    using parent_type = Tensor;

    /// Pull in types from parent_type
    using pimpl_pointer          = typename parent_type::pimpl_pointer;
    using logical_layout_type    = typename parent_type::logical_layout_type;
    using logical_layout_pointer = typename parent_type::logical_layout_pointer;
    using buffer_pointer         = typename parent_type::buffer_pointer;

    /** @brief Value constructor.
     *
     *  @param[in] plogical A pointer to the logical layout of *this.
     *  @param[in] pbuffer A pointer to the memory storing the elements of
     *                     *this.
     *
     *  @throw std::runtime_error if @p plogical or @p pbuffer is a nullptr.
     *                            Strong throw guarantee.
     */
    TensorPIMPL(logical_layout_pointer plogical, buffer_pointer pbuffer);

    /** @brief Initializes *this to a deep copy of @p other.
     *
     *  @param[in] other The TensorPIMPL to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copies.
     *                        Strong throw guarantee.
     */
    TensorPIMPL(const TensorPIMPL& other) :
      m_plogical_(other.m_plogical_->clone_as<logical_layout_type>()),
      m_pbuffer_(other.m_pbuffer_->clone()) {}

    /** @brief Returns a deep copy of *this.
     *
     *  @return A pointer to a deep copy of *this.
     *
     *  @throw std::bad_alloc if there is a problem copying *this. Strong throw
     *                        guarantee.
     */
    pimpl_pointer clone() const { return std::make_unique<TensorPIMPL>(*this); }

    // -------------------------------------------------------------------------
    // -- Accessors
    // -------------------------------------------------------------------------

    /// Provides mutable access to the logical layout
    auto& logical_layout() { return *m_plogical_; }

    /// Provides read-only access to the logical layout
    const auto& logical_layout() const { return *m_plogical_; }

    /// Provides mutable access to the buffer.
    auto& buffer() { return *m_pbuffer_; }

    /// Provides read-only access to the buffer
    const auto& buffer() const { return *m_pbuffer_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two TensorPIMPL objects are value equal if their respective logical
     *  layouts and buffers compare value equal. Value equality of the layout
     *  and the buffer are done polymorphically.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const TensorPIMPL& rhs) const noexcept {
        if(m_plogical_->are_different(*rhs.m_plogical_)) return false;
        return m_pbuffer_->are_equal(*rhs.m_pbuffer_);
    }

private:
    /// How users will think of *this
    logical_layout_pointer m_plogical_;

    /// The literal elements of the tensor
    buffer_pointer m_pbuffer_;
};

inline TensorPIMPL::TensorPIMPL(logical_layout_pointer plogical,
                                buffer_pointer pbuffer) :
  m_plogical_(std::move(plogical)), m_pbuffer_(std::move(pbuffer)) {
    if(m_plogical_ == nullptr)
        throw std::runtime_error("Logical layout should not be null.");
    if(m_pbuffer_ == nullptr)
        throw std::runtime_error("Buffer should not be null");
}

} // namespace tensorwrapper::detail_
