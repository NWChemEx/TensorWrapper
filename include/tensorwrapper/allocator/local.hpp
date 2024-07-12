#pragma once
#include <tensorwrapper/allocator/allocator_base.hpp>

namespace tensorwrapper::allocator {

/** @brief Can create buffers that exist entirely in local memory.
 *
 *  This class is presently a stub that will be filled in later, as needed.
 */
class Local : public AllocatorBase {
private:
    /// Type *this inherits from
    using my_base_type = AllocatorBase;

public:
    // Pull in base's ctors
    using my_base_type::my_base_type;
};

} // namespace tensorwrapper::allocator
