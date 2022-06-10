/* Functions, types, and includes common to the unit tests focusing on testing
 * the tensor component of Libtensorwrapper.
 */
#pragma once
#include "../buffer/make_pimpl.hpp"
#include "shapes/make_tot_shape.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensordetail_/pimpl.hpp"
#include "tensorwrapper/tensortensor.hpp"
#include <catch2/catch.hpp>
#include <utilities/type_traits/variant/cat.hpp>

namespace testing {

/// Function which generates some dummy tensors for a given type
template<typename FieldType>
auto get_tensors() {
    using namespace tensorwrapper::tensor::novel;
    using namespace tensorwrapper::tensor;
    using tensor_type  = TensorWrapper<FieldType>;
    using pimpl_type   = detail_::TensorWrapperPIMPL<FieldType>;
    using shape_type   = typename tensor_type::shape_type;
    using extents_type = typename tensor_type::extents_type;
    using buffer_type  = typename tensor_type::buffer_type;
    std::map<std::string, tensor_type> rv;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        auto [vec_bp, mat_bp, t3d_bp] =
          make_pimpl<FieldType>(); // Create Buffers
        auto vec_b = std::make_unique<buffer_type>(vec_bp->clone());
        auto mat_b = std::make_unique<buffer_type>(mat_bp->clone());
        auto t3d_b = std::make_unique<buffer_type>(t3d_bp->clone());

        auto vec_shape = std::make_unique<shape_type>(extents_type{3});
        auto mat_shape = std::make_unique<shape_type>(extents_type{2, 2});
        auto t3d_shape = std::make_unique<shape_type>(extents_type{2, 2, 2});
        auto palloc    = default_allocator<FieldType>();

        auto vec_p = std::make_unique<pimpl_type>(
          std::move(vec_b), std::move(vec_shape), palloc->clone());
        auto mat_p = std::make_unique<pimpl_type>(
          std::move(mat_b), std::move(mat_shape), palloc->clone());
        auto t3d_p = std::make_unique<pimpl_type>(
          std::move(t3d_b), std::move(t3d_shape), palloc->clone());

        rv["vector"] = tensor_type(std::move(vec_p));
        rv["matrix"] = tensor_type(std::move(mat_p));
        rv["tensor"] = tensor_type(std::move(t3d_p));
    } else {
        auto [vov_bp, vom_bp, mov_bp] =
          make_pimpl<FieldType>(); // Create Buffers

        auto vov_b = std::make_unique<buffer_type>(vov_bp->clone());
        auto vom_b = std::make_unique<buffer_type>(vom_bp->clone());
        auto mov_b = std::make_unique<buffer_type>(mov_bp->clone());

        extents_type vector_extents{3};
        extents_type matrix_extents{2, 2};

        auto vov_shape =
          testing::make_uniform_tot_shape(vector_extents, vector_extents);
        auto vom_shape =
          testing::make_uniform_tot_shape(vector_extents, matrix_extents);
        auto mov_shape =
          testing::make_uniform_tot_shape(matrix_extents, vector_extents);
        auto palloc = default_allocator<FieldType>();

        auto vov_p = std::make_unique<pimpl_type>(
          std::move(vov_b), vov_shape.clone(), palloc->clone());
        auto vom_p = std::make_unique<pimpl_type>(
          std::move(vom_b), vom_shape.clone(), palloc->clone());
        auto mov_p = std::make_unique<pimpl_type>(
          std::move(mov_b), mov_shape.clone(), palloc->clone());

        rv["vector-of-vectors"]  = tensor_type(std::move(vov_p));
        rv["vector-of-matrices"] = tensor_type(std::move(vom_p));
        rv["matrix-of-vectors"]  = tensor_type(std::move(mov_p));
    }
    return rv;
}

} // namespace testing
