#pragma once
#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {

class Local : public BufferBase {
private:
    using base_type = BufferBase;

public:
    // Pull in base's ctors
    using base_type::base_type;
};

} // namespace tensorwrapper::buffer
