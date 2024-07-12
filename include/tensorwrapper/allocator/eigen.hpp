#pragma once
#include <tensorwrapper/allocator/replicated.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/mono_tile.hpp>

namespace tensorwrapper::allocator {

/** @brief Used to allocate buffers which rely on Eigen tensors.
 *
 *  @tparam FloatType
 */
template<typename FloatType, unsigned short Rank>
class Eigen : public Replicated {
private:
    /// The type of *this
    using my_type = Eigen<FloatType, Rank>;

    /// The class *this inherits from
    using my_base_type = Replicated;

public:
    // Pull in base class's types
    using my_base_type::base_pointer;
    using my_base_type::buffer_base_pointer;
    using my_base_type::const_base_reference;
    using my_base_type::layout_pointer;

    /// Type of a buffer containing an Eigen tensor
    using eigen_buffer_type = buffer::Eigen<FloatType, Rank>;

    /// Type of a pointer to an eigen_buffer_type object
    using eigen_buffer_pointer = std::unique_ptr<eigen_buffer_type>;

    /// Type of a layout which can be used to create an Eigen tensor
    using eigen_layout_type = layout::MonoTile;

    /// Type of a read-only reference to an object of type eigen_layout_type
    using const_eigen_layout_reference = const eigen_layout_type&;

    /// Type of a pointer to an eigen_layout_type object
    using eigen_layout_pointer = std::unique_ptr<eigen_layout_type>;

    // Reuse base class's ctors
    using my_base_type::my_base_type;

    eigen_buffer_pointer allocate(const_eigen_layout_reference layout) {
        return allocate(std::make_unique<eigen_layout_type>(layout));
    }

    eigen_buffer_pointer allocate(eigen_layout_pointer playout);

protected:
    buffer_base_pointer allocate_(layout_pointer playout) override;

    base_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }

    bool are_equal_(const_base_reference rhs) const noexcept override {
        return my_base_type::are_equal_impl_<my_type>(rhs);
    }
};

// -----------------------------------------------------------------------------
// -- Explicit class template declarations
// -----------------------------------------------------------------------------

#define DECLARE_EIGEN_ALLOCATOR(RANK)         \
    extern template class Eigen<float, RANK>; \
    extern template class Eigen<double, RANK>

DECLARE_EIGEN_ALLOCATOR(0);
DECLARE_EIGEN_ALLOCATOR(1);
DECLARE_EIGEN_ALLOCATOR(2);
DECLARE_EIGEN_ALLOCATOR(3);
DECLARE_EIGEN_ALLOCATOR(4);
DECLARE_EIGEN_ALLOCATOR(5);
DECLARE_EIGEN_ALLOCATOR(6);
DECLARE_EIGEN_ALLOCATOR(7);
DECLARE_EIGEN_ALLOCATOR(8);
DECLARE_EIGEN_ALLOCATOR(9);
DECLARE_EIGEN_ALLOCATOR(10);

#undef DECLARE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
