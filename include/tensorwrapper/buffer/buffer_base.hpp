#pragma once
#include <tensorwrapper/layout/tiled.hpp>

namespace tensorwrapper::buffer {

class BufferBase {
public:
    /// Type of the class describing the physical layout of the buffer
    using layout_type = layout::Tiled;

    /// Type of a read-only reference to a layout
    using const_layout_reference = const layout_type&;

    /// Type of a pointer to the layout
    using layout_pointer = typename layout_type::layout_pointer;

    BufferBase() noexcept = default;

    explicit BufferBase(const_layout_reference layout) :
      BufferBase(layout.clone()) {}

    explicit BufferBase(layout_pointer playout) noexcept :
      m_layout_(std::move(playout)) {}

private:
    /// The layout of *this
    layout_pointer m_layout_;
};

} // namespace tensorwrapper::buffer
