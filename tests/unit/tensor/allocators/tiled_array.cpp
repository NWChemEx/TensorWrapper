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
#include "tensorwrapper/tensor/allocator/allocator.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

TEMPLATE_TEST_CASE("TiledArrayAllocator Generic", "[allocator][ta]",
                   field::Scalar, field::Tensor) {
    using field_type = TestType;
    using alloc_type = allocator::TiledArrayAllocator<field_type>;
    using allocator::ta::Distribution;
    using allocator::ta::Storage;

    alloc_type defaulted;
    alloc_type non_default(Storage::Core, Distribution::Distributed);

    SECTION("Default State") {
        REQUIRE(defaulted.storage() == Storage::Core);
        REQUIRE(defaulted.dist() == Distribution::Replicated);
    }

    SECTION("Manual Ctor") {
        REQUIRE(non_default.storage() == Storage::Core);
        REQUIRE(non_default.dist() == Distribution::Distributed);

        REQUIRE(defaulted != non_default);
        REQUIRE_FALSE(defaulted == non_default);
    }

    SECTION("Copy Ctor") {
        alloc_type copy(defaulted);
        REQUIRE(defaulted == copy);
        REQUIRE_FALSE(defaulted != copy);
    }

    SECTION("Move Ctor") {
        alloc_type other_defaulted;
        alloc_type copy(std::move(other_defaulted));
        REQUIRE(defaulted == copy);
        REQUIRE_FALSE(defaulted != copy);
    }

    SECTION("is_equal") {
        SECTION("Both Default") {
            alloc_type other_defaulted;
            REQUIRE(defaulted.is_equal(other_defaulted));
            REQUIRE(other_defaulted.is_equal(defaulted));
        }

        SECTION("Different Specs") {
            REQUIRE_FALSE(non_default.is_equal(defaulted));
            REQUIRE_FALSE(defaulted.is_equal(non_default));
        }
    }

    SECTION("Clone") {
        auto copy = defaulted.clone();
        REQUIRE(copy->is_equal(defaulted));
    }
}

TEST_CASE("TiledArrayAllocator<Scalar>") {
    using field_type     = field::Scalar;
    using buffer_type    = buffer::Buffer<field_type>;
    using allocator_type = allocator::TiledArrayAllocator<field_type>;

    using ta_trange_type = TA::TiledRange;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;

    using extents_type = typename allocator_type::extents_type;
    using tiling_type  = typename allocator_type::tiling_type;
    using shape_type   = typename allocator_type::shape_type;

    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();

    SECTION("OneBigTile") {
        shape_type vec_shape(extents_type{3});
        shape_type mat_shape(extents_type{2, 2});
        shape_type ten_shape(extents_type{2, 2, 2});

        // Default tiling is OneBigTile
        buffer_type vec(pvec->clone());
        buffer_type mat(pmat->clone());
        buffer_type ten(pt3d->clone());

        allocator_type alloc(Storage::Core);

        SECTION("allocate(rank 1) - tile op") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
                REQUIRE(extent == 3);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, vec_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(*buf == vec);
        }

        SECTION("allocate(rank 1) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](std::vector<size_t> idx) -> double {
                element_count++; // Count the number of invocations
                REQUIRE(idx.size() == 1);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 3);
                return idx[0] + 1;
            };

            auto buf = alloc.allocate(fxn, vec_shape);
            REQUIRE(element_count == 3);
            REQUIRE(*buf == vec);
        }

        SECTION("allocate(rank 2) - tile op") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 2);
                REQUIRE(up.size() == 2);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                REQUIRE(extent_0 == 2);
                REQUIRE(extent_1 == 2);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j) {
                        data[i * extent_1 + j] =
                          (i + lo[0]) * 2 + (j + lo[1]) + 1;
                    }
            };

            auto buf = alloc.allocate(fxn, mat_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(*buf == mat);
        }

        SECTION("allocate(rank 2) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](std::vector<size_t> idx) -> double {
                element_count++; // Count the number of invocations
                REQUIRE(idx.size() == 2);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 2);
                REQUIRE(idx[1] >= 0);
                REQUIRE(idx[1] < 2);
                return 2 * idx[0] + idx[1] + 1;
            };

            auto buf = alloc.allocate(fxn, mat_shape);
            REQUIRE(element_count == 4);
            REQUIRE(*buf == mat);
        }

        SECTION("allocate(rank 3) - tile op") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++;
                REQUIRE(lo.size() == 3);
                REQUIRE(up.size() == 3);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                REQUIRE(lo[2] >= 0);
                REQUIRE(up[2] <= 2);
                REQUIRE(lo[2] < up[2]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                size_t extent_2 = up[2] - lo[2];
                REQUIRE(extent_0 == 2);
                REQUIRE(extent_1 == 2);
                REQUIRE(extent_2 == 2);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j)
                        for(auto k = 0; k < extent_2; ++k) {
                            data[i * extent_1 * extent_2 + j * extent_2 + k] =
                              (i + lo[0]) * 4 + (j + lo[1]) * 2 + (k + lo[2]) +
                              1;
                        }
            };

            auto buf = alloc.allocate(fxn, ten_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(*buf == ten);
        }

        SECTION("allocate(rank 3) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](std::vector<size_t> idx) -> double {
                element_count++;
                REQUIRE(idx.size() == 3);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 2);
                REQUIRE(idx[1] >= 0);
                REQUIRE(idx[1] < 2);
                REQUIRE(idx[2] >= 0);
                REQUIRE(idx[2] < 2);
                return 4 * idx[0] + 2 * idx[1] + idx[2] + 1;
            };

            auto buf = alloc.allocate(fxn, ten_shape);
            REQUIRE(element_count == 8);
            REQUIRE(*buf == ten);
        }
    }

    SECTION("SingleElementTile") {
        shape_type vec_shape(tiling_type{{0, 1, 2, 3}});
        shape_type mat_shape(tiling_type{{0, 1, 2}, {0, 1, 2}});
        shape_type ten_shape(tiling_type{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}});

        // Default tiling is OneBigTile, retile to SingleElementTile
        ta_trange_type se_tr_vec{{0, 1, 2, 3}};
        ta_trange_type se_tr_mat{{0, 1, 2}, {0, 1, 2}};
        ta_trange_type se_tr_ten{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
        pvec->retile(se_tr_vec);
        pmat->retile(se_tr_mat);
        pt3d->retile(se_tr_ten);
        buffer_type vec(pvec->clone());
        buffer_type mat(pmat->clone());
        buffer_type ten(pt3d->clone());

        allocator_type alloc(Storage::Core);

        SECTION("allocate(rank 1)") {
            std::atomic<int> inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
                REQUIRE(extent == 1);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, vec_shape);
            REQUIRE(inner_tile_count == 3); // One Tile For Each Element
            REQUIRE(*buf == vec);
        }

        SECTION("allocate(rank 2)") {
            std::atomic<int> inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 2);
                REQUIRE(up.size() == 2);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                REQUIRE(extent_0 == 1);
                REQUIRE(extent_1 == 1);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j) {
                        data[i * extent_1 + j] =
                          (i + lo[0]) * 2 + (j + lo[1]) + 1;
                    }
            };

            auto buf = alloc.allocate(fxn, mat_shape);
            REQUIRE(inner_tile_count == 4); // One Tile For Each Element
            REQUIRE(*buf == mat);
        }

        SECTION("allocate(rank 3)") {
            std::atomic<int> inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++;
                REQUIRE(lo.size() == 3);
                REQUIRE(up.size() == 3);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                REQUIRE(lo[2] >= 0);
                REQUIRE(up[2] <= 2);
                REQUIRE(lo[2] < up[2]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                size_t extent_2 = up[2] - lo[2];
                REQUIRE(extent_0 == 1);
                REQUIRE(extent_1 == 1);
                REQUIRE(extent_2 == 1);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j)
                        for(auto k = 0; k < extent_2; ++k) {
                            data[i * extent_1 * extent_2 + j * extent_2 + k] =
                              (i + lo[0]) * 4 + (j + lo[1]) * 2 + (k + lo[2]) +
                              1;
                        }
            };

            auto buf = alloc.allocate(fxn, ten_shape);
            REQUIRE(inner_tile_count == 8); // One Tile For Each Element
            REQUIRE(*buf == ten);
        }
    }
}

TEST_CASE("TiledArrayAllocator<Tensor>") {
    using field_type     = field::Tensor;
    using buffer_type    = buffer::Buffer<field_type>;
    using allocator_type = allocator::TiledArrayAllocator<field_type>;

    using ta_trange_type = TA::TiledRange;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;

    using extents_type = typename allocator_type::extents_type;
    using shape_type   = typename allocator_type::shape_type;

    auto&& [pvov, pvom, pmov] = testing::make_pimpl<field_type>();

    extents_type vector_extents = {3};
    extents_type matrix_extents = {2, 2};
    auto vov_shape =
      testing::make_uniform_tot_shape(vector_extents, vector_extents);
    auto vom_shape =
      testing::make_uniform_tot_shape(vector_extents, matrix_extents);
    auto mov_shape =
      testing::make_uniform_tot_shape(matrix_extents, vector_extents);

    SECTION("OneBigTile") {
        // Default tiling is OneBigTile
        buffer_type vov(pvov->clone());
        buffer_type vom(pvom->clone());
        buffer_type mov(pmov->clone());

        allocator_type alloc(Storage::Core);

        SECTION("allocate(vov) - tile op") {
            size_t outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
                REQUIRE(extent == 3);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, vov_shape);
            REQUIRE(outer_tile_count == 3);
            REQUIRE(*buf == vov);
        }

        SECTION("allocate(vov) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](auto outer, auto idx) -> double {
                element_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(idx.size() == 1);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 3);
                REQUIRE(outer[0] >= 0);
                REQUIRE(outer[0] < 3);
                return idx[0] + 1;
            };

            auto buf = alloc.allocate(fxn, vov_shape);
            REQUIRE(element_count == 9);
            REQUIRE(*buf == vov);
        }

        SECTION("allocate(vom) - tile op") {
            size_t outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(lo.size() == 2);
                REQUIRE(up.size() == 2);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                REQUIRE(extent_0 == 2);
                REQUIRE(extent_1 == 2);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j) {
                        data[i * extent_1 + j] =
                          (i + lo[0]) * 2 + (j + lo[1]) + 1;
                    }
            };

            auto buf = alloc.allocate(fxn, vom_shape);
            REQUIRE(outer_tile_count == 3);
            REQUIRE(*buf == vom);
        }

        SECTION("allocate(vom) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](auto outer, auto idx) -> double {
                element_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(outer[0] >= 0);
                REQUIRE(outer[0] < 3);
                REQUIRE(idx.size() == 2);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 2);
                REQUIRE(idx[1] >= 0);
                REQUIRE(idx[1] < 2);
                return 2 * idx[0] + idx[1] + 1;
            };

            auto buf = alloc.allocate(fxn, vom_shape);
            REQUIRE(element_count == 12);
            REQUIRE(*buf == vom);
        }

        SECTION("allocate(mov) - tile op") {
            size_t outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 2);
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
                REQUIRE(extent == 3);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, mov_shape);
            REQUIRE(outer_tile_count == 4);
            REQUIRE(*buf == mov);
        }

        SECTION("allocate(mov) - scalar op") {
            size_t element_count = 0;
            auto fxn             = [&](auto outer, auto idx) -> double {
                element_count++;
                REQUIRE(outer.size() == 2);
                REQUIRE(idx.size() == 1);
                REQUIRE(idx[0] >= 0);
                REQUIRE(idx[0] < 3);
                REQUIRE(outer[0] >= 0);
                REQUIRE(outer[0] < 2);
                REQUIRE(outer[1] >= 0);
                REQUIRE(outer[1] < 2);
                return idx[0] + 1;
            };

            auto buf = alloc.allocate(fxn, mov_shape);
            REQUIRE(element_count == 12);
            REQUIRE(*buf == mov);
        }
    } // OneBigTile

    SECTION("SingleElementTile") {
#if 0 // Enable when retile for ToT is implemented
      // Default tiling is OneBigTile, retile to SingleElementTile
	ta_trange_type  se_tr_vec{{0,1,2,3}};
	ta_trange_type  se_tr_mat{{0,1,2},{0,1,2}};
	pvov->retile(se_tr_vec);
	pvom->retile(se_tr_vec);
	pmov->retile(se_tr_mat);
        buffer_type vov(pvov->clone());
        buffer_type vom(pvom->clone());
        buffer_type mov(pmov->clone());

        allocator_type alloc(Storage::Core, Tiling::SingleElementTile);

        SECTION("allocate(vov)") {
            std::atomic<int> outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
		REQUIRE(extent == 3);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, vov_shape);
            REQUIRE(outer_tile_count == 3);
            REQUIRE(*buf == vov);
        }

        SECTION("allocate(vom)") {
            std::atomic<int> outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 1);
                REQUIRE(lo.size() == 2);
                REQUIRE(up.size() == 2);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
		REQUIRE(extent_0 == 2);
		REQUIRE(extent_1 == 2);
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j) {
                        data[i * extent_1 + j] = (i + lo[0]) * 2 + (j + lo[1]) + 1;
                    }
            };

            auto buf = alloc.allocate(fxn, vom_shape);
            REQUIRE(outer_tile_count == 3);
            REQUIRE(*buf == vom);
        }

        SECTION("allocate(mov)") {
            std::atomic<int> outer_tile_count = 0;
            auto fxn = [&](std::vector<size_t> outer, std::vector<size_t> lo,
                           std::vector<size_t> up, double* data) {
                outer_tile_count++;
                REQUIRE(outer.size() == 2);
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
		REQUIRE(extent == 3);
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, mov_shape);
            REQUIRE(outer_tile_count == 4);
            REQUIRE(*buf == mov);
        }
#endif
    } // SingleElementtile
}
