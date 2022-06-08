#pragma once
#include "tensorwrapper/tensor/novel/shapes/shapes.hpp"
namespace testing {

namespace detail {
inline auto global_to_local(size_t idx, const std::vector<size_t>& dims) {
    std::vector<size_t> local_idx(dims.size());
    for(auto i = 0; i < dims.size() - 1; ++i) {
        const auto d = dims[i];
        local_idx[i] = idx % d;
        idx          = idx / d;
    }
    local_idx[dims.size() - 1] = idx;
    return local_idx;
}
} // namespace detail

inline auto make_uniform_tot_map(const std::vector<size_t>& outer_extents,
                                 const std::vector<size_t>& inner_extents) {
    using namespace tensorwrapper::sparse_map;
    using namespace tensorwrapper::tensor::novel;
    using namespace tensorwrapper::tensor;
    Shape<field::Scalar> inner_shape(inner_extents);
    std::map<Index, Shape<field::Scalar>> inner_map;

    size_t volume = 1;
    for(auto i : outer_extents) volume *= i;
    for(auto global_idx = 0ul; global_idx < volume; ++global_idx) {
        auto _idx = detail::global_to_local(global_idx, outer_extents);
        Index idx(_idx.begin(), _idx.end());
        inner_map[idx] = inner_shape;
    }

    return inner_map;
}

template<typename RetT = tensorwrapper::tensor::novel::Shape<
           tensorwrapper::tensor::field::Tensor>>
auto make_uniform_tot_shape(const std::vector<size_t>& outer_extents,
                            const std::vector<size_t>& inner_extents) {
    return RetT(outer_extents,
                make_uniform_tot_map(outer_extents, inner_extents));
}

} // namespace testing
