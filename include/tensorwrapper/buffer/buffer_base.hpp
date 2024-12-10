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
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/dsl/labeled.hpp>
#include <tensorwrapper/layout/layout_base.hpp>

namespace tensorwrapper::buffer {

/** @brief Common base class for all buffer objects.
 *
 *  All classes which wrap existing tensor libraries derive from this class.
 */
class BufferBase : public detail_::PolymorphicBase<BufferBase> {
private:
    /// Type of *this
    using my_type = BufferBase;

    /// Type *this inherits from
    using my_base_type = detail_::PolymorphicBase<my_type>;

public:
    /// Type all buffers inherit from
    using buffer_base_type = typename my_base_type::base_type;

    /// Type of a mutable reference to a buffer_base_type object
    using buffer_base_reference = typename my_base_type::base_reference;

    /// Type of a read-only reference to a buffer_base_type object
    using const_buffer_base_reference =
      typename my_base_type::const_base_reference;

    /// Type of a pointer to an object of type buffer_base_type
    using buffer_base_pointer = typename my_base_type::base_pointer;

    /// Type of a pointer to a read-only object of type buffer_base_type
    using const_buffer_base_pointer = typename my_base_type::const_base_pointer;

    /// Type of the class describing the physical layout of the buffer
    using layout_type = layout::LayoutBase;

    /// Type of a read-only reference to a layout
    using const_layout_reference = const layout_type&;

    /// Type of a pointer to the layout
    using layout_pointer = typename layout_type::layout_pointer;

    /// Type of labels for making a labeled buffer
    using label_type = std::string;

    /// Type of a labeled buffer
    using labeled_buffer_type = dsl::Labeled<buffer_base_type, label_type>;

    /// Type of a labeled read-only buffer (n.b. labels are mutable)
    using labeled_const_buffer_type = dsl::Labeled<const buffer_base_type>;

    /// Type of a read-only reference to a labeled_buffer_type object
    using const_labeled_buffer_reference = const labeled_buffer_type&;

    // -------------------------------------------------------------------------
    // -- Accessors
    // -------------------------------------------------------------------------

    /** @brief Does *this have a layout?
     *
     *  Default constructed or moved from BufferBase objects do not have
     *  layouts. This method is used to determine if *this has a layout or not.
     *
     *  @return True if *this has a layout and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool has_layout() const noexcept { return static_cast<bool>(m_layout_); }

    /** @brief Retrieves the layout of *this.
     *
     *  This method can be used to retrieve the layout associated with *this,
     *  assuming there is one. See has_layout for determining if *this has a
     *  layout or not.
     *
     *  @return A read-only reference to the layout.
     *
     *  @throw std::runtime_error if *this does not have a layout. Strong throw
     *                            guarantee.
     */
    const_layout_reference layout() const {
        assert_layout_();
        return *m_layout_;
    }

    // -------------------------------------------------------------------------
    // -- BLAS Operations
    // -------------------------------------------------------------------------

    buffer_base_reference addition_assignment(
      label_type this_labels, const_labeled_buffer_reference rhs) {
        return addition_assignment_(std::move(this_labels), rhs);
    }

    buffer_base_pointer addition(label_type this_labels,
                                 const_labeled_buffer_reference rhs) const {
        auto pthis = clone();
        pthis->addition_assignment(std::move(this_labels), rhs);
        return pthis;
    }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Associates labels with the modes of *this.
     *
     *  This method is used to create a labeled buffer object by pairing *this
     *  with the provided labels. The resulting object is capable of being
     *  composed via the DSL.
     *
     *  @param[in] labels The indices to associate with the modes of *this.
     *
     *  @return A DSL term pairing *this with @p labels.
     *
     *  @throw None No throw guarantee.
     */
    labeled_buffer_type operator()(label_type labels);

    /** @brief Associates labels with the modes of *this.
     *
     *  This method is the same as the non-const version except that the result
     *  contains a read-only reference to *this.
     *
     *  @param[in] labels The labels to associate with *this.
     *
     *  @return A DSL term pairing *this with @p labels.
     *
     *  @throw None No throw guarantee.
     */
    labeled_const_buffer_type operator()(label_type labels) const;

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two BufferBase objects are value equal if the layouts they contain are
     *  polymorphically value equal or if both BufferBase objects do not contain
     *  a layout.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const BufferBase& rhs) const noexcept {
        if(has_layout() != rhs.has_layout()) return false;
        if(!has_layout()) return true;
        return m_layout_->are_equal(*rhs.m_layout_);
    }

    /** @brief Is *this different from @p rhs?
     *
     *  This method defines "different from" as being "not value equal." See
     *  the description of operator== for the definition of value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */

    bool operator!=(const BufferBase& rhs) const noexcept {
        return !(*this == rhs);
    }

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
      BufferBase(layout.clone()) {}

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
      m_layout_(other.m_layout_ ? other.m_layout_->clone() : nullptr) {}

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
            auto temp = rhs.has_layout() ? rhs.m_layout_->clone() : nullptr;
            temp.swap(m_layout_);
        }
        return *this;
    }

    /// Derived class should overwrite to implement addition_assignment
    virtual buffer_base_reference addition_assignment_(
      label_type this_labels, const_labeled_buffer_reference rhs) {
        throw std::runtime_error("Addition assignment NYI");
    }

private:
    /// Throws std::runtime_error when there is no layout
    void assert_layout_() const {
        if(has_layout()) return;
        throw std::runtime_error(
          "Buffer has no layout. Was it default initialized?");
    }

    /// The layout of *this
    layout_pointer m_layout_;
};

} // namespace tensorwrapper::buffer
