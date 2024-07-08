#pragma once
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/layout/tiled.hpp>
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

    /// Type of a read-only reference to a buffer_base_type object
    using const_buffer_base_reference =
      typename my_base_type::const_base_reference;

    /// Type of a pointer to an object of type buffer_base_type
    using buffer_base_pointer = typename my_base_type::base_pointer;

    /// Type of the class describing the physical layout of the buffer
    using layout_type = layout::Tiled;

    /// Type of a read-only reference to a layout
    using const_layout_reference = const layout_type&;

    /// Type of a pointer to the layout
    using layout_pointer = typename layout_type::layout_pointer;

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
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two BufferBase objects are value equal if the layouts they contain are
     *  polymorphically value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const BufferBase& rhs) const noexcept {
        return m_layout_->are_equal(*rhs.m_layout_);
    }

    bool operator!=(const BufferBase& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    // -------------------------------------------------------------------------
    // -- Ctors, assignment
    // -------------------------------------------------------------------------

    /** @brief Creates a buffer with no layout.
     *
     *  @throw None No throw guarantee.
     */
    BufferBase() : BufferBase(nullptr) {}

    explicit BufferBase(const_layout_reference layout) :
      BufferBase(layout.clone()) {}

    explicit BufferBase(layout_pointer playout) noexcept :
      m_layout_(std::move(playout)) {}

    BufferBase(const BufferBase& other) : m_layout_(other.m_layout_->clone()) {}

    BufferBase& operator=(const BufferBase& rhs) {
        if(this != &rhs) {
            auto temp = rhs.has_layout() ? rhs.m_layout_->clone() : nullptr;
            temp.swap(m_layout_);
        }
        return *this;
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