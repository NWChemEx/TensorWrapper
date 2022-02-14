#pragma once
#include "tensorwrapper/detail_/hashing.hpp"
#include "tensorwrapper/tensor/fields.hpp"
#include "tensorwrapper/tensor/shapes/shape.hpp"
#include <memory>
#include <string>

namespace tensorwrapper::tensor::buffer {
namespace detail_ {
template<typename FieldType>
class BufferPIMPL;

}

/// For the moment just used to set types for the PIMPLs
template<typename FieldType>
class Buffer {
private:
    /// Type of the PIMPL
    using pimpl_type = detail_::BufferPIMPL<FieldType>;

public:
    /// Type used for indices in einsum/index-based operations
    using annotation_type = std::string;

    /// Type of a read-only reference to an annotation
    using const_annotation_reference = const std::string&;

    /// Type of a pointer to the PIMPL
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /// Type used to model the shape
    using shape_type = Shape<FieldType>;

    /// Type of a read-only reference to the shape
    using const_shape_reference = const shape_type&;

    /// Type of the object used for hashing
    using hasher_type = tensorwrapper::detail_::Hasher;

    /// Mutable reference to a hasher
    using hasher_reference = hasher_type&;
};

} // namespace tensorwrapper::tensor::buffer
