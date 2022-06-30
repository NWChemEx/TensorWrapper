#pragma once
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor {

template<typename FieldType>
class TensorWrapper;

template<typename FieldType>
class Shape;

namespace allocator {

template<typename FieldType>
class Allocator;

}

namespace buffer {

template<typename FieldType>
class Buffer;

}

namespace expression {

template<typename FieldType>
class LabeledView;

}

namespace field {

template<typename FieldType>
struct FieldTraits {
    using label_type = std::string;

    using const_label_reference = const label_type&;

    using tensor_type = TensorWrapper<FieldType>;

    using allocator_type = allocator::Allocator<FieldType>;

    using const_allocator_reference = const allocator_type&;

    using allocator_pointer = std::unique_ptr<allocator_type>;

    using shape_type = Shape<FieldType>;

    using const_shape_reference = const shape_type&;

    using shape_pointer = std::unique_ptr<shape_type>;

    using buffer_type    = buffer::Buffer<FieldType>;
    using buffer_pointer = std::unique_ptr<buffer_type>;

    using labeled_tensor = expression::LabeledView<FieldType>;
};

} // namespace field
} // namespace tensorwrapper::tensor
