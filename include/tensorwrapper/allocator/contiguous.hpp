#pragma once
#include <tensorwrapper/allocator/replicated.hpp>
#include <tensorwrapper/types/il_traits.hpp>

namespace tensorwrapper::allocator {

/** @brief Allocator that can create Contiguous buffers.
 *
 *  @tparam FloatType Type of the elements in the contiguous buffer.
 */
template<typename FloatType>
class Contiguous : public Replicated {
private:
    /// Type of *this
    using my_type = Contiguous<FloatType>;

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

    /// Type of the buffer associated with *this
    using contiguous_buffer_type = buffer::Contiguous<element_type>;
    using contiguous_pointer     = std::unique_ptr<contiguous_buffer_type>;

    /// Type of initializer lists
    using rank0_il = typename types::ILTraits<element_type, 0>::type;
    using rank1_il = typename types::ILTraits<element_type, 1>::type;
    using rank2_il = typename types::ILTraits<element_type, 2>::type;
    using rank3_il = typename types::ILTraits<element_type, 3>::type;
    using rank4_il = typename types::ILTraits<element_type, 4>::type;

    /// Pull in base class's ctors
    using base_type::base_type;

    /** @brief Allocates a contiguous pointer given @p layout.
     *
     *  @note This method shadows the similar function in the base class. The
     *        intent is to avoid needing to rebind a freshly allocated buffer
     *         when the user already knows it is a Contiguous buffer
     */
    contiguous_pointer allocate(const_layout_reference layout) {
        return allocate(layout.clone_as<layout_type>());
    }
    contiguous_pointer allocate(layout_pointer layout) {
        auto p = allocate_(std::move(layout));
        return detail_::static_pointer_cast<contiguous_buffer_type>(p);
    }

    contiguous_pointer construct(rank0_il il) { return construct_(il); }
    contiguous_pointer construct(rank1_il il) { return construct_(il); }
    contiguous_pointer construct(rank2_il il) { return construct_(il); }
    contiguous_pointer construct(rank3_il il) { return construct_(il); }
    contiguous_pointer construct(rank4_il il) { return construct_(il); }

    contiguous_pointer construct(layout_pointer layout, element_type value) {
        return construct_(std::move(layout), std::move(value));
    }

    contiguous_pointer construct(const_layout_reference layout,
                                 element_type value) {
        return construct(layout.clone(), std::move(value));
    }

protected:
    virtual contiguous_pointer construct_(rank0_il il) = 0;
    virtual contiguous_pointer construct_(rank1_il il) = 0;
    virtual contiguous_pointer construct_(rank2_il il) = 0;
    virtual contiguous_pointer construct_(rank3_il il) = 0;
    virtual contiguous_pointer construct_(rank4_il il) = 0;

    /// To be overridden by the derived class to implement construct
    virtual contiguous_pointer construct_(layout_pointer layout,
                                          element_type value) = 0;
};

} // namespace tensorwrapper::allocator