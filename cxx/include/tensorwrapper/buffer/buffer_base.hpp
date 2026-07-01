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
#include <tensorwrapper/buffer/buffer_base_common.hpp>
#include <tensorwrapper/detail_/dsl_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/dsl/labeled.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/types/buffer_traits.hpp>

namespace tensorwrapper::buffer {

/** @brief Common base class for all buffer objects.
 *
 *  All classes which own their state and wrap existing tensor libraries derive
 *  from this class.
 */
class BufferBase : public BufferBaseCommon<BufferBase>,
                   public tensorwrapper::detail_::PolymorphicBase<BufferBase>,
                   public tensorwrapper::detail_::DSLBase<BufferBase> {
private:
    /// Type of *this
    using my_type = BufferBase;

    /// Type of the common base class
    using common_base = BufferBaseCommon<my_type>;

    /// Traits of my_type
    using my_traits = types::ClassTraits<my_type>;

protected:
    /// Type *this inherits from
    using polymorphic_base = tensorwrapper::detail_::PolymorphicBase<my_type>;

public:
    /// Type all buffers inherit from
    using buffer_base_type = typename my_traits::buffer_base_type;

    /// Type of a reference to an object of type buffer_base_type
    using buffer_base_reference = typename my_traits::buffer_base_reference;

    /// Type of a reference to a read-only object of type buffer_base_type
    using const_buffer_base_reference =
      typename my_traits::const_buffer_base_reference;

    /// Type of a pointer to an object of type buffer_base_type
    using buffer_base_pointer = typename my_traits::buffer_base_pointer;

    /// Type of a pointer to a read-only object of type buffer_base_type
    using const_buffer_base_pointer =
      typename my_traits::const_buffer_base_pointer;

    /// Type of a pointer to the layout
    using layout_pointer = std::unique_ptr<layout_type>;

protected:
    // -------------------------------------------------------------------------
    // -- Ctors, assignment
    // -------------------------------------------------------------------------

    /** @brief Creates a buffer with no layout.
     *
     *  This ctor is protected because users should not directly construct
     *  BufferBase objects. BufferBase objects are always created by derived
     *  classes.
     *
     *  @throw None No throw guarantee.
     */
    BufferBase() : BufferBase(nullptr) {}

    /** @brief Creates a buffer initialized with a copy of @p layout.
     *
     *  @param[in] layout The physical layout of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy of
     *                        @p layout. Strong throw guarantee.
     */
    explicit BufferBase(const_layout_reference layout) :
      BufferBase(layout.clone_as<layout_type>()) {}

    /** @brief Creates a buffer which owns the layout pointed to by @p playout.
     *
     *  @param[in] playout A pointer to the layout for *this.
     *
     *  @throw None No throw guarantee.
     */

    explicit BufferBase(layout_pointer playout) noexcept :
      m_layout_(std::move(playout)) {}

    /** @brief Creates a buffer by deep copying @p other.
     *
     *  @param[in] other The buffer to copy.
     *
     *  @throw std::bad_alloc if there is a problem copying the state of
     *                        @p other. Strong throw guarantee.
     */
    BufferBase(const BufferBase& other) :
      m_layout_(other.m_layout_ ? other.m_layout_->clone_as<layout_type>() :
                                  nullptr) {}

    /** @brief Replaces the state in *this with a deep copy of the state in
     *         @p rhs.
     *
     *  @param[in] rhs The buffer to copy the state of.
     *
     *  @return *this after replacing its state with a copy of @p rhs.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    BufferBase& operator=(const BufferBase& rhs) {
        if(this != &rhs) {
            auto temp_layout = rhs.has_layout() ?
                                 rhs.m_layout_->clone_as<layout_type>() :
                                 nullptr;

            temp_layout.swap(m_layout_);
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // -- BufferBaseCommon hooks
    // -------------------------------------------------------------------------
    friend common_base;
    bool has_layout_() const noexcept { return static_cast<bool>(m_layout_); }

    const_layout_reference layout_() const { return *m_layout_; }

    layout_reference layout_() { return *m_layout_; }

    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;

    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;

    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

    virtual bool approximately_equal_(const BufferBase& rhs,
                                      double tol) const = 0;

    template<typename BufferBaseType>
    bool approximately_equal_(const BufferViewBase<BufferBaseType>& rhs,
                              double tol) const;

private:
    template<typename FxnType>
    dsl_reference binary_op_common_(FxnType&& fxn, label_type this_labels,
                                    const_labeled_reference lhs,
                                    const_labeled_reference rhs);

    /// The layout of *this
    layout_pointer m_layout_;
};

} // namespace tensorwrapper::buffer
