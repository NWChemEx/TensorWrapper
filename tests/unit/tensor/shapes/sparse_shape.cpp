/*
 * Copyright 2022 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "make_tot_shape.hpp"
#include "tensorwrapper/tensor/allocator/allocator.hpp"
#include "tensorwrapper/tensor/shapes/shapes.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

/* Testing Strategy:
 *
 * We assume that the PIMPL works and that the base class works. We need to test
 * that:
 * - the ctor forwards arguments to PIMPL correctly
 * - clone behaves polymorphically
 * - the tensor returned is correct
 * - comparisons
 */
TEST_CASE("SparseShape<field::Scalar>") {
    using field_type      = field::Scalar;
    using shape_type      = SparseShape<field_type>;
    using extents_type    = typename shape_type::extents_type;
    using tiling_type     = typename shape_type::tiling_type;
    using sparse_map_type = typename shape_type::sparse_map_type;
    using idx_type        = typename sparse_map_type::key_type;
    using idx2mode_type   = typename shape_type::idx2mode_type;

    extents_type matrix_extents{3, 4};
    extents_type tensor_extents{2, 2, 2};

    tiling_type matrix_tiling{{0, 3}, {0, 4}};

    idx_type i0{0}, i1{1}, i00{0, 0}, i11{1, 1};
    sparse_map_type matrix_sm{{i0, {i0}}, {i1, {i1}}};
    sparse_map_type tensor_sm{{i00, {i0}}, {i11, {i1}}};

    idx2mode_type i2m{1, 0};
    idx2mode_type i2m1{1, 2, 0};

    shape_type m(matrix_extents, matrix_sm);
    shape_type mt(matrix_extents, matrix_sm, i2m);
    shape_type t(tensor_extents, tensor_sm);
    shape_type tt(tensor_extents, tensor_sm, i2m1);

    shape_type m2(matrix_tiling, matrix_sm);
    shape_type m2t(matrix_tiling, matrix_sm, i2m);

    SECTION("CTors") {
        SECTION("No idx2mode") {
            REQUIRE(m.extents() == matrix_extents);
            REQUIRE(t.extents() == tensor_extents);

            REQUIRE(m == m2);

            // Make sure there's not an extra copy
            auto pm = matrix_extents.data();
            shape_type m2(std::move(matrix_extents), matrix_sm);
            REQUIRE(m2.extents().data() == pm);

            // Throws if sm is inconsistent with extents
            REQUIRE_THROWS_AS(shape_type(matrix_extents, tensor_sm),
                              std::runtime_error);
        }

        SECTION("idx2mode") {
            REQUIRE(mt.extents() == matrix_extents);
            REQUIRE(tt.extents() == tensor_extents);

            REQUIRE(mt == m2t);

            // Make sure there's not an extra copy
            auto pm = matrix_extents.data();
            shape_type m2(std::move(matrix_extents), matrix_sm, i2m);
            REQUIRE(m2.extents().data() == pm);

            // Throws if sm is inconsistent with idx2mode
            REQUIRE_THROWS_AS(shape_type(matrix_extents, matrix_sm, i2m1),
                              std::runtime_error);

            // Throws if an element of idx2mode is out of range
            extents_type i2m2{0, 2};
            REQUIRE_THROWS_AS(shape_type(matrix_extents, matrix_sm, i2m2),
                              std::runtime_error);
        }
    }

#if 0
    SECTION("make_tensor") {
        using ta_tensor = TA::Tensor<float>;
        using ta_shape  = TA::SparseShape<float>;

        SingleElementTiles<field_type> a;
        auto max = std::numeric_limits<float>::max();

        SECTION("Matrix") {
            auto matrix  = m.make_tensor(a);
            auto corr_tr = a.make_tiled_range(matrix_extents);

            ta_tensor shape_data(TA::Range(3, 4),
                                 {max, 0, 0, 0, 0, max, 0, 0, 0, 0, 0, 0});
            ta_shape corr_shape(shape_data, corr_tr);
            REQUIRE(std::get<0>(matrix).trange() == corr_tr);
            REQUIRE(std::get<0>(matrix).shape() == corr_shape);
        }

        SECTION("Tensor") {
            auto tensor  = t.make_tensor(a);
            auto corr_tr = a.make_tiled_range(tensor_extents);

            ta_tensor shape_data(TA::Range(2, 2, 2),
                                 {max, 0, 0, 0, 0, 0, 0, max});
            ta_shape corr_shape(shape_data, corr_tr);
            REQUIRE(std::get<0>(tensor).trange() == corr_tr);
            REQUIRE(std::get<0>(tensor).shape() == corr_shape);
        }
    }
#endif

    SECTION("comparisons") {
        // Same
        REQUIRE(m == shape_type(matrix_extents, matrix_sm));
        REQUIRE_FALSE(m != shape_type(matrix_extents, matrix_sm));
        auto other_inner_map =
          testing::make_uniform_tot_map(matrix_extents, {{1}});

        // Different fields
        using other_shape_type = SparseShape<field::Tensor>;
        REQUIRE_FALSE(
          m == other_shape_type(matrix_extents, other_inner_map, tensor_sm));
        REQUIRE(m !=
                other_shape_type(matrix_extents, other_inner_map, tensor_sm));

        // Different extents
        REQUIRE_FALSE(m == shape_type(extents_type{5, 5}, matrix_sm));
        REQUIRE(m != shape_type(extents_type{5, 5}, matrix_sm));

        // Different sparse maps
        sparse_map_type sm2{{i0, {i0, i1}}, {i1, {i0, i1}}};
        REQUIRE_FALSE(m == shape_type(matrix_extents, sm2));
        REQUIRE(m != shape_type(matrix_extents, sm2));

        // Different permutation
        REQUIRE_FALSE(m == mt);
        REQUIRE(m != mt);

        // Base class's operator== is non-polymorphic
        using base_type = Shape<field_type>;
        base_type b(matrix_extents);
        REQUIRE(b == static_cast<base_type&>(m));
        REQUIRE_FALSE(b != static_cast<base_type&>(m));
    }
}

TEST_CASE("SparseShape<field::Tensor>") {
    using field_type         = field::Tensor;
    using shape_type         = SparseShape<field_type>;
    using extents_type       = typename shape_type::extents_type;
    using inner_extents_type = typename shape_type::inner_extents_type;
    using sparse_map_type    = typename shape_type::sparse_map_type;
    using idx_type           = typename sparse_map_type::key_type;
    using idx2mode_type      = typename shape_type::idx2mode_type;

    extents_type extents{3, 4};
    extents_type inner_extents{50, 203};
    auto inner_map = testing::make_uniform_tot_map(extents, inner_extents);

    idx_type i0{0}, i1{1}, i00{0, 0}, i11{1, 1};
    sparse_map_type sm{{i00, {i0}}, {i11, {i1}}};

    idx2mode_type i2m{1, 0};

    shape_type t(extents, inner_map, sm);
    shape_type tt(extents, inner_map, sm, i2m);

    SECTION("CTors") {
        SECTION("No idx2mode") {
            REQUIRE(t.extents() == extents);

            // Make sure there's not an extra copy
            auto pm = extents.data();
            // auto ipm = inner_extents.data();
            decltype(inner_map) inner_map_cpy = inner_map;
            shape_type m2(std::move(extents), std::move(inner_map_cpy), sm);
            REQUIRE(m2.extents().data() == pm);
            // REQUIRE(m2.inner_extents().data() == ipm);

            sparse_map_type sm2{{i0, {i0}}, {i1, {i1}}};
            // Throws if sm is inconsistent with extents
            REQUIRE_THROWS_AS(shape_type(extents, inner_map, sm2),
                              std::runtime_error);
        }

        SECTION("idx2mode") {
            REQUIRE(tt.extents() == extents);

            // Make sure there's not an extra copy
            auto pm = extents.data();
            // auto ipm = inner_extents.data();
            decltype(inner_map) inner_map_cpy = inner_map;
            shape_type m2(std::move(extents), std::move(inner_map_cpy), sm,
                          i2m);
            REQUIRE(m2.extents().data() == pm);
            // REQUIRE(m2.inner_extents().data() == ipm);

            // Throws if sm is inconsistent with idx2mode
            REQUIRE_THROWS_AS(
              shape_type(extents, inner_map, sm, idx2mode_type{1, 2, 0}),
              std::runtime_error);

            // Throws if an element of idx2mode is out of range
            extents_type i2m2{0, 5};
            REQUIRE_THROWS_AS(shape_type(extents, inner_map, sm, i2m2),
                              std::runtime_error);
        }
    }

#if 0
    SECTION("make_tensor") {
        using ta_tensor = TA::Tensor<float>;
        using ta_shape  = TA::SparseShape<float>;

        SingleElementTiles<field_type> a;
        auto max = std::numeric_limits<float>::max();

        SECTION("Matrix") {
            auto matrix  = t.make_tensor(a);
            auto corr_tr = a.make_tiled_range(extents);

            ta_tensor shape_data(TA::Range(3, 4),
                                 {max, 0, 0, 0, 0, max, 0, 0, 0, 0, 0, 0});
            ta_shape corr_shape(shape_data, corr_tr);
            REQUIRE(std::get<0>(matrix).trange() == corr_tr);
            REQUIRE(std::get<0>(matrix).shape() == corr_shape);
        }
    }
#endif

    SECTION("comparisons") {
        // Same
        REQUIRE(t == shape_type(extents, inner_map, sm));
        REQUIRE_FALSE(t != shape_type(extents, inner_map, sm));

        // Different fields
        using other_shape_type = SparseShape<field::Scalar>;
        REQUIRE_FALSE(t == other_shape_type(extents_type{3, 4, 5}, sm));
        REQUIRE(t != other_shape_type(extents_type{3, 4, 5}, sm));

        // Different extents
        REQUIRE_FALSE(t == shape_type(extents_type{5, 5}, inner_map, sm));
        REQUIRE(t != shape_type(extents_type{5, 5}, inner_map, sm));

        // Different sparse maps
        sparse_map_type sm2{{i00, {i0, i1}}, {i11, {i0, i1}}};
        REQUIRE_FALSE(t == shape_type(extents, inner_map, sm2));
        REQUIRE(t != shape_type(extents, inner_map, sm2));

        // Different permutation
        REQUIRE_FALSE(t == tt);
        REQUIRE(t != tt);

        // Base class's operator== is non-polymorphic
        using base_type = Shape<field_type>;
        base_type b(extents, inner_map);
        REQUIRE(b == t);
        REQUIRE_FALSE(b != t);
    }
}
