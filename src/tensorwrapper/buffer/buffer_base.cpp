#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {

typename BufferBase::labeled_buffer_type BufferBase::operator()(
  label_type labels) {
    return labeled_buffer_type(*this, std::move(labels));
}

typename BufferBase::labeled_const_buffer_type BufferBase::operator()(
  label_type labels) const {
    return labeled_const_buffer_type(*this, std::move(labels));
}

} // namespace tensorwrapper::buffer