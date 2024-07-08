#pragma once
#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {

/** @brief Establishes that the state in the buffer is obtainable without
 *         communication.
 *
 *  For now this class is a strong type and does not impart any additional state
 *  to the BufferBase class.
 *
 */
class Local : public BufferBase {
private:
    /// Type *this inherits from
    using my_base_type = BufferBase;

public:
    // Pull in base's ctors
    using my_base_type::my_base_type;
};

} // namespace tensorwrapper::buffer
