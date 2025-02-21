#pragma once
#include <memory>
#include <tensorwrapper/buffer/buffer_fwd.hpp>
#include <tensorwrapper/types/class_traits.hpp>

namespace tensorwrapper::types {

template<>
struct ClassTraits<buffer::BufferBase> {
    /// Type all buffers inherit from
    using buffer_base_type = buffer::BufferBase;

    /// Type of a mutable reference to a buffer_base_type object
    using buffer_base_reference = buffer_base_type&;

    /// Type of a read-only reference to a buffer_base_type object
    using const_buffer_base_reference = const buffer_base_type&;

    /// Type of a unique_ptr to a mutable buffer_base_type
    using buffer_base_pointer = std::unique_ptr<buffer_base_type>;

    /// Type of a unique_ptr to a mutable buffer_base_type
    using const_buffer_base_pointer = std::unique_ptr<const buffer_base_type>;
};

} // namespace tensorwrapper::types