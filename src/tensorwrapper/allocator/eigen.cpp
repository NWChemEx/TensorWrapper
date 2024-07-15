#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::allocator {
namespace {
template<typename EigenTensorType, typename ShapeType, std::size_t... Is>
auto unwrap_shape(const ShapeType& shape, std::index_sequence<Is...>) {
    // XXX: This is a hack until we have a general Shape API in place
    auto const_shape = static_cast<const shape::Smooth&>(shape);
    return EigenTensorType(const_shape.extent(Is)...);
}

} // namespace

#define TPARAMS template<typename FloatType, unsigned short Rank>
#define EIGEN Eigen<FloatType, Rank>

TPARAMS
typename EIGEN::eigen_buffer_pointer EIGEN::allocate(
  eigen_layout_pointer playout) {
    using eigen_tensor_type = typename eigen_buffer_type::tensor_type;
    if(playout->shape().rank() != Rank)
        throw std::runtime_error("Rank of the layout is not compatible");

    return std::make_unique<eigen_buffer_type>(
      unwrap_shape<eigen_tensor_type>(playout->shape(),
                                      std::make_index_sequence<Rank>()),
      *playout);
}

// -----------------------------------------------------------------------------
// -- Protected methods
// -----------------------------------------------------------------------------

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::allocate_(layout_pointer playout) {
    auto pderived = detail_::dynamic_pointer_cast<eigen_layout_type>(playout);
    if(pderived == nullptr) throw std::runtime_error("Unsupported layout");

    return allocate(std::move(pderived));
}

#undef EIGEN
#undef TPARAMS

// -- Explicit class template instantiation

#define DEFINE_EIGEN_ALLOCATOR(RANK)   \
    template class Eigen<float, RANK>; \
    template class Eigen<double, RANK>

DEFINE_EIGEN_ALLOCATOR(0);
DEFINE_EIGEN_ALLOCATOR(1);
DEFINE_EIGEN_ALLOCATOR(2);
DEFINE_EIGEN_ALLOCATOR(3);
DEFINE_EIGEN_ALLOCATOR(4);
DEFINE_EIGEN_ALLOCATOR(5);
DEFINE_EIGEN_ALLOCATOR(6);
DEFINE_EIGEN_ALLOCATOR(7);
DEFINE_EIGEN_ALLOCATOR(8);
DEFINE_EIGEN_ALLOCATOR(9);
DEFINE_EIGEN_ALLOCATOR(10);

#undef DEFINE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
