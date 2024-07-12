#pragma once
#include <tensorwrapper/allocator/local.hpp>

namespace tensorwrapper::allocator {

/** @brief Can create buffers that exist entirely in local memory and are
 *         guaranteed to be the same for all processes.
 *
 *  This class is presently a stub that will be filled in later, as needed.
 */
class Replicated : public Local {
private:
    /// Type *this inherits from
    using my_base_type = Local;

public:
    // Pull in base's ctors
    using my_base_type::my_base_type;
};

} // namespace tensorwrapper::allocator
