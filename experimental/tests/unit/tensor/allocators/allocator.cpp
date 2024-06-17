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

#include "../buffer/make_pimpl.hpp"
#include "../shapes/make_tot_shape.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/allocator/allocator.hpp"
#include <catch2/catch.hpp>

/* Testing Strategy:
 *
 * The unit tests focuse on the parts of the Allocator hierarchy which are
 * implemented in the Allocator class (as opposed to the derived classes).
 * Unit tests for the derived classes focus on the parts they implement.
 */

using namespace tensorwrapper::tensor;

TEMPLATE_TEST_CASE("Allocator Generic", "[allocator][generic]", field::Scalar,
                   field::Tensor) {
    using field_type = TestType;
    auto palloc      = default_allocator<field_type>();

    SECTION("Comparisons") {
        const auto prhs = default_allocator<field_type>();
        REQUIRE(*palloc == *prhs);
        REQUIRE_FALSE(*palloc != *prhs);

        REQUIRE(palloc->is_equal(*prhs));
    }

    SECTION("Clone") {
        const auto copy = palloc->clone();
        REQUIRE(*palloc == *copy);
        REQUIRE_FALSE(*palloc != *copy);
        REQUIRE(palloc->is_equal(*copy));
    }
}

TEST_CASE("Allocator<Scalar>") {
    using field_type  = field::Scalar;
    using buffer_type = buffer::Buffer<field_type>;
    auto palloc       = default_allocator<field_type>();

    using allocator_type = typename decltype(palloc)::element_type;
    using extents_type   = typename allocator_type::extents_type;
    using shape_type     = typename allocator_type::shape_type;

    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();
    SECTION("allocate(rank 1 tensor)") {
        buffer_type vec(pvec->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
            size_t extent = up[0] - lo[0];
            for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
        };

        extents_type vec_extents{3};
        shape_type vec_shape(vec_extents);
        auto buf = palloc->allocate(fxn, vec_shape);
        REQUIRE(*buf == vec);
    }

    SECTION("allocate(rank 2 tensor)") {
        buffer_type mat(pmat->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
            size_t extent_0 = up[0] - lo[0];
            size_t extent_1 = up[1] - lo[1];
            for(auto i = 0; i < extent_0; ++i)
                for(auto j = 0; j < extent_1; ++j) {
                    data[i * extent_1 + j] = (i + lo[0]) * 2 + (j + lo[1]) + 1;
                }
        };

        extents_type mat_extents{2, 2};
        shape_type mat_shape(mat_extents);
        auto buf = palloc->allocate(fxn, mat_shape);
        REQUIRE(*buf == mat);
    }

    SECTION("allocate(rank 3 tensor)") {
        buffer_type ten3(pt3d->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
            size_t extent_0 = up[0] - lo[0];
            size_t extent_1 = up[1] - lo[1];
            size_t extent_2 = up[2] - lo[2];
            for(auto i = 0; i < extent_0; ++i)
                for(auto j = 0; j < extent_1; ++j)
                    for(auto k = 0; k < extent_2; ++k) {
                        data[i * extent_1 * extent_2 + j * extent_2 + k] =
                          (i + lo[0]) * 4 + (j + lo[1]) * 2 + (k + lo[2]) + 1;
                    }
        };

        extents_type ten_extents{2, 2, 2};
        shape_type ten_shape(ten_extents);
        auto buf = palloc->allocate(fxn, ten_shape);
        REQUIRE(*buf == ten3);
    }
}

TEST_CASE("Allocator<Tensor>") {
    using field_type  = field::Tensor;
    using buffer_type = buffer::Buffer<field_type>;
    auto palloc       = default_allocator<field_type>();

    using allocator_type = typename decltype(palloc)::element_type;
    using extents_type   = typename allocator_type::extents_type;
    using shape_type     = typename allocator_type::shape_type;

    auto&& [pvov, pvom, pmov]   = testing::make_pimpl<field_type>();
    extents_type vector_extents = {3};
    extents_type matrix_extents = {2, 2};
    auto vov_shape =
      testing::make_uniform_tot_shape(vector_extents, vector_extents);
    auto vom_shape =
      testing::make_uniform_tot_shape(vector_extents, matrix_extents);
    auto mov_shape =
      testing::make_uniform_tot_shape(matrix_extents, vector_extents);

    SECTION("allocate(vov)") {
        buffer_type vov(pvov->clone());
        auto fxn = [](std::vector<size_t> outer, std::vector<size_t> lo,
                      std::vector<size_t> up, double* data) {
            size_t extent = up[0] - lo[0];
            for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
        };

        auto buf = palloc->allocate(fxn, vov_shape);
        REQUIRE(*buf == vov);
    }

    SECTION("allocate(vom)") {
        buffer_type vom(pvom->clone());
        auto fxn = [](std::vector<size_t> outer, std::vector<size_t> lo,
                      std::vector<size_t> up, double* data) {
            size_t extent_0 = up[0] - lo[0];
            size_t extent_1 = up[1] - lo[1];
            for(auto i = 0; i < extent_0; ++i)
                for(auto j = 0; j < extent_1; ++j) {
                    data[i * extent_1 + j] = (i + lo[0]) * 2 + (j + lo[1]) + 1;
                }
        };

        auto buf = palloc->allocate(fxn, vom_shape);
        REQUIRE(*buf == vom);
    }

    SECTION("allocate(mov)") {
        buffer_type mov(pmov->clone());
        auto fxn = [](std::vector<size_t> outer, std::vector<size_t> lo,
                      std::vector<size_t> up, double* data) {
            size_t extent = up[0] - lo[0];
            for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
        };

        auto buf = palloc->allocate(fxn, mov_shape);
        REQUIRE(*buf == mov);
    }
}
