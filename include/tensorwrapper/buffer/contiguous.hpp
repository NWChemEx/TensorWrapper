#pragma once
#include <tensorwrapper/buffer/replicated.hpp>

namespace tensorwrapper::buffer {

template<typename FloatType>
class Contiguous : public Replicated {
public:
    /// Type of the floating-point values in *this
    using float_type = FloatType;

    /// Type of a mutable pointer to the buffer
    using pointer = float_type*;

    /// Type of a read-only pointer to the buffer
    using const_pointer = const float_type*;

    /// Pull in base class's ctors
    using Replicated::Replicated;

    /** @brief A pointer to the 0-th element in the buffer.
     *
     *  @return A pointer to the 0-th element in the buffer.
     *
     *  @throw ??? Throws if the derived class throws. Same throw guarantee.
     */
    pointer data() { return data_(); }

    /** @brief Returns a read-only pointer to the 0-th element in the buffer.
     *
     *  @return A read-only pointer to the 0-th element in the buffer.
     *
     *  @throw ??? Throws if the derived class throws. Same throw guarantee.
     */
    const_pointer data() const { return data_(); }

    /** @brief Returns the number of elements in the contiguous buffer.
     *
     *  @return The length of the contiguous buffer.
     *
     *  @throw None No throw guarantee.
     */
    auto size() const noexcept { return layout.shape().size(); }

protected:
    /// Derived class should overwrite consistent with data()
    virtual pointer data_() = 0;

    /// Derived class should overwrite consistent with data()const
    virtual const_pointer data_() const = 0;
};

} // namespace tensorwrapper::buffer