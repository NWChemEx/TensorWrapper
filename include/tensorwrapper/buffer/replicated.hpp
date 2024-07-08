#pragma once
#include <tensorwrapper/buffer/local.hpp>

namespace tensorwrapper::buffer {

/** @brief Denotes that a buffer is the same on all processes.
 *
 *  At the moment this class is a strong type and has no additional state over
 *  its base class.
 */
class Replicated : public Local {
private:
    /// Type *this derives from
    using my_base_type = Local;

public:
    // Pull in base's ctors
    using my_base_type::my_base_type;
};

} // namespace tensorwrapper::buffer
