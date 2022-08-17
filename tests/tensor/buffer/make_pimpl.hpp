#pragma once
#include "tensorwrapper/ta_helpers/lazy_tile.hpp"
#include "tensorwrapper/tensor/buffer/detail_/ta_buffer_pimpl.hpp"

namespace testing {

template<typename FieldType>
auto make_pimpl() {
    namespace tensor = tensorwrapper::tensor;

    using field_type    = FieldType;
    using buffer_type   = tensor::buffer::detail_::TABufferPIMPL<field_type>;
    using tensor_type   = typename buffer_type::default_tensor_type;
    using trange_type   = typename buffer_type::ta_trange_type;
    using ta_shape_type = typename buffer_type::ta_shape_type;

    auto& world = TA::get_default_world();
    if constexpr(std::is_same_v<field_type, tensor::field::Scalar>) {
        tensor_type vec_ta(world, {1.0, 2.0, 3.0});
        tensor_type mat_ta(world, {{1.0, 2.0}, {3.0, 4.0}});
        tensor_type t3d_ta(
          world, {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});

        auto vec = std::make_unique<buffer_type>(vec_ta);
        auto mat = std::make_unique<buffer_type>(mat_ta);
        auto t3d = std::make_unique<buffer_type>(t3d_ta);
        return std::make_tuple(std::move(vec), std::move(mat), std::move(t3d));
    } else {
        using tile_type  = typename tensor_type::value_type;
        using inner_tile = typename tile_type::value_type;
        inner_tile v0(TA::Range{3}, {1.0, 2.0, 3.0});
        inner_tile m0(TA::Range{2, 2}, {1.0, 2.0, 3.0, 4.0});

        tensor_type vov_ta(world, {v0, v0, v0});
        tensor_type vom_ta(world, {m0, m0, m0});
        tensor_type mov_ta(world, {{v0, v0}, {v0, v0}});

        auto vov = std::make_unique<buffer_type>(vov_ta);
        auto vom = std::make_unique<buffer_type>(vom_ta);
        auto mov = std::make_unique<buffer_type>(mov_ta);
        return std::make_tuple(std::move(vov), std::move(vom), std::move(mov));
    }
}

inline auto make_direct_pimpl() {
    namespace tensor = tensorwrapper::tensor;

    using field_type  = tensor::field::Scalar;
    using buffer_type = tensor::buffer::detail_::TABufferPIMPL<field_type>;
    using tensor_type = typename buffer_type::lazy_tensor_type;
    using trange_type = typename buffer_type::ta_trange_type;

    using lazy_tile_type = tensorwrapper::ta_helpers::lazy_scalar_type;
    using range_type     = typename lazy_tile_type::range_type;
    using tile_type      = typename lazy_tile_type::eval_type;

    auto scalar_lambda = [](range_type range) -> tile_type {
        auto t = tile_type(range, 0.0);
        for(const auto& idx : range) {
            auto n_dims  = idx.size();
            double value = 1.0;
            for(auto i = 0; i < n_dims; ++i) {
                value += std::pow(2.0, i) * idx[n_dims - 1 - i];
            }
            t[idx] = value;
        }
        return t;
    };
    lazy_tile_type::add_evaluator(scalar_lambda, "scalar_test");

    auto tile_lambda = [](lazy_tile_type& t, const range_type& r) -> float {
        t = lazy_tile_type(r, "scalar_test");
        return 1.0;
    };

    auto& world = TA::get_default_world();
    auto vec_ta = TiledArray::make_array<tensor_type>(
      world, trange_type{{0, 3}}, tile_lambda);
    auto mat_ta = TiledArray::make_array<tensor_type>(
      world, trange_type{{0, 2}, {0, 2}}, tile_lambda);
    auto t3d_ta = TiledArray::make_array<tensor_type>(
      world, trange_type{{0, 2}, {0, 2}, {0, 2}}, tile_lambda);

    auto vec = std::make_unique<buffer_type>(vec_ta);
    auto mat = std::make_unique<buffer_type>(mat_ta);
    auto t3d = std::make_unique<buffer_type>(t3d_ta);
    return std::make_tuple(std::move(vec), std::move(mat), std::move(t3d));
}

} // namespace testing
