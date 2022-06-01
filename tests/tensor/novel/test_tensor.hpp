/* Functions, types, and includes common to the unit tests focusing on testing
 * the tensor component of Libtensorwrapper.
 */
#pragma once
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/novel/tensor.hpp"
#include "tensorwrapper/tensor/novel/detail_/pimpl.hpp"
#include "../buffer/make_pimpl.hpp"
#include <catch2/catch.hpp>
#include <utilities/type_traits/variant/cat.hpp>

namespace testing {

/// Function which generates some dummy tensors for a given type
template<typename FieldType>
auto get_tensors() {
    using namespace tensorwrapper::tensor::novel;
    using namespace tensorwrapper::tensor;
    using tensor_type = novel::TensorWrapper<FieldType>;
    using pimpl_type  = novel::detail_::TensorWrapperPIMPL<FieldType>;
    using shape_type  = typename tensor_type::shape_type;
    using extents_type = typename tensor_type::extents_type;
    using buffer_type  = typename tensor_type::buffer_type;
    std::map<std::string, tensor_type> rv;
    if constexpr(field::is_scalar_field_v<FieldType>) {
	auto [vec_bp, mat_bp, t3d_bp] = make_pimpl<FieldType>(); // Create Buffers
	auto vec_b = std::make_unique<buffer_type>(vec_bp->clone());
	auto mat_b = std::make_unique<buffer_type>(mat_bp->clone());
	auto t3d_b = std::make_unique<buffer_type>(t3d_bp->clone());

        auto vec_shape = std::make_unique<shape_type>(extents_type{3});
	auto mat_shape = std::make_unique<shape_type>(extents_type{2,2});
	auto t3d_shape = std::make_unique<shape_type>(extents_type{2,2,2});
	auto palloc = default_allocator<FieldType>();

	auto vec_p = std::make_unique<pimpl_type>( std::move(vec_b), std::move(vec_shape), palloc->clone() );
	auto mat_p = std::make_unique<pimpl_type>( std::move(mat_b), std::move(mat_shape), palloc->clone() );
	auto t3d_p = std::make_unique<pimpl_type>( std::move(t3d_b), std::move(t3d_shape), palloc->clone() );


        rv["vector"] = tensor_type(std::move(vec_p));
        rv["matrix"] = tensor_type(std::move(mat_p));
        rv["tensor"] = tensor_type(std::move(t3d_p));
    } else {
#if 0
        using outer_tile = typename TensorType::value_type;
        using inner_tile = typename outer_tile::value_type;
        using dvector_il = TA::detail::vector_il<double>;
        using vector_il  = TA::detail::vector_il<inner_tile>;
        using matrix_il  = TA::detail::matrix_il<inner_tile>;

        inner_tile v0(TA::Range({2}), {1.0, 2.0});
        inner_tile v1(TA::Range({2}), {3.0, 4.0});
        inner_tile v2(TA::Range({2}), {5.0, 6.0});
        inner_tile v3(TA::Range({2}), {7.0, 8.0});
        inner_tile mat0(TA::Range({2, 2}), dvector_il{1.0, 2.0, 3.0, 4.0});
        inner_tile mat1(TA::Range({2, 2}), dvector_il{5.0, 6.0, 7.0, 8.0});
        rv["vector-of-vectors"] = TensorType(world, vector_il{v0, v1});
        rv["matrix-of-vectors"] =
          TensorType(world, matrix_il{vector_il{v0, v1}, vector_il{v2, v3}});
        rv["vector-of-matrices"] = TensorType(world, vector_il{mat0, mat1});
#endif
    }
    return rv;
}

} // namespace testing
