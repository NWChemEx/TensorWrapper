#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/utilities/floating_point_dispatch.hpp>

namespace tensorwrapper::operations {
namespace {
struct InfinityKernel {
    template<typename FloatType>
    Tensor run(const buffer::BufferBase& t) {
        using allocator_type = allocator::Eigen<FloatType>;
        allocator_type alloc(t.allocator().runtime());
        FloatType max_element{0.0};
        const auto& buffer_down = alloc.rebind(t);
        for(std::size_t i = 0; i < buffer_down.size(); ++i) {
            auto elem = types::fabs(*(buffer_down.data() + i));
            if(elem > max_element) max_element = elem;
        }
        shape::Smooth s{};
        layout::Physical l(s);
        auto pbuffer = alloc.construct(l, max_element);
        return Tensor(s, std::move(pbuffer));
    }
};

} // namespace

Tensor infinity_norm(const Tensor& t) {
    InfinityKernel k;
    return utilities::floating_point_dispatch(k, t.buffer());
}

} // namespace tensorwrapper::operations