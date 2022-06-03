#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/conversion/conversion.hpp"
#include <catch2/catch.hpp>

/// TA Types
template<typename TileType>
using distarray_t = TA::DistArray<TileType, TA::SparsePolicy>;
using tat_t       = distarray_t<TA::Tensor<double>>;
using tot_t       = distarray_t<TA::Tensor<TA::Tensor<double>>>;
using tile_t      = typename tot_t::value_type;
using inner_t     = typename tile_t::value_type;

/// Conversion Types
using s_conversion_t = tensorwrapper::tensor::Conversion<tat_t>;
using t_conversion_t = tensorwrapper::tensor::Conversion<tot_t>;

/// Field Types
using s_t = tensorwrapper::tensor::field::Scalar;
using t_t = tensorwrapper::tensor::field::Tensor;

/// Buffer Types
using s_buffer_t   = tensorwrapper::tensor::buffer::Buffer<s_t>;
using t_buffer_t   = tensorwrapper::tensor::buffer::Buffer<t_t>;
using s_tabuffer_t = tensorwrapper::tensor::buffer::detail_::TABufferPIMPL<s_t>;
using t_tabuffer_t = tensorwrapper::tensor::buffer::detail_::TABufferPIMPL<t_t>;

using tensorwrapper::ta_helpers::allclose;
using tensorwrapper::ta_helpers::allclose_tot;

TEST_CASE("Conversion") {
    /// Make the tensors to wrap
    auto& world = TA::get_default_world();
    inner_t v0(TA::Range{3}, {1.0, 2.0, 3.0});
    tat_t corr_mat(world, {{1.0, 2.0}, {3.0, 4.0}});
    tot_t corr_vov(world, {v0, v0, v0});

    /// Make buffers wrapping the tensors
    auto s_tabuffer = std::make_unique<s_tabuffer_t>(corr_mat);
    auto t_tabuffer = std::make_unique<t_tabuffer_t>(corr_vov);
    s_buffer_t s_buffer(s_tabuffer->clone());
    t_buffer_t t_buffer(t_tabuffer->clone());

    /// Instances of conversion
    s_conversion_t s_conversion;
    t_conversion_t t_conversion;

    SECTION("convert") {
        /// Scalar check
        REQUIRE(allclose(s_conversion.convert(s_buffer), corr_mat));
        /// ToT check
        REQUIRE(allclose_tot(t_conversion.convert(t_buffer), corr_vov, 1));
    }

    SECTION("can_convert") {
        REQUIRE(s_conversion.can_convert(s_buffer));
        REQUIRE(t_conversion.can_convert(t_buffer));
        REQUIRE_FALSE(s_conversion.can_convert(t_buffer));
        REQUIRE_FALSE(t_conversion.can_convert(s_buffer));
    }
}
