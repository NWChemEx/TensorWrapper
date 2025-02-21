#pragma once
#include <tensorwrapper/allocator/replicated.hpp>

namespace tensorwrapper::allocator {

/** @brief Allocator that can create Contiguous buffers.
 *
 *  @tparam FloatType Type of the elements in the contiguous buffer.
 */
template<typename FloatType>
class Contiguous : public Replicated {
private:
    /// Type *this derives from
    using base_type = Replicated;

public:
    /// Pull in base types
    ///@{
    using base_type::buffer_base_pointer;
    using base_type::const_layout_reference;
    ///@}

    /// Type of each element in the tensor
    using element_type = FloatType;

    /// Pull in base class's ctors
    using base_type::base_type;

    buffer_base_pointer construct(layout_pointer layout, element_type value) {
        return construct_(std::move(layout), std::move(value));
    }

    buffer_base_pointer construct(const_layout_reference layout,
                                  element_type value) {
        return construct(layout.clone(), std::move(value));
    }

protected:
    /// To be overridden by the derived class to implement construct
    virtual buffer_base_pointer construct_(layout_pointer layout,
                                           element_type value) = 0;
};

} // namespace tensorwrapper::allocator