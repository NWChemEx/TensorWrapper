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
#include "tensorwrapper/ta_helpers/slice.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/detail_/pimpl.hpp"
#include <catch2/catch.hpp>

namespace ta_helpers = tensorwrapper::ta_helpers;
using namespace tensorwrapper::tensor;

/* Testing Strategy:
 *
 * We assume that all allocators and shapes work correctly. this means that
 * functions which depend on the shape and allocator state should work correctly
 * as long as those functions properly call and process the results of
 * interacting with allocators/shapes. In practice there are a lot of
 */

TEST_CASE("TensorWrapperPIMPL<Tensor>") {
    using field_type     = field::Tensor;
    using pimpl_type     = detail_::TensorWrapperPIMPL<field_type>;
    using buffer_type    = typename pimpl_type::buffer_type;
    using buffer_pointer = typename pimpl_type::buffer_pointer;
    using shape_type     = typename pimpl_type::shape_type;
    using extents_type   = typename pimpl_type::extents_type;
    using ta_trange_type = TA::TiledRange;
    using ta_tensor_type =
      TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;

    auto palloc = default_allocator<field_type>();
    auto oalloc = allocator::ta_allocator<field_type>(
      Storage::Core, Distribution::Distributed);

    buffer_pointer vov_buffer_obt, vom_buffer_obt, mov_buffer_obt;
    // TODO: Test SET
    // buffer_pointer vov_buffer_set, vom_buffer_set, mov_buffer_set;
    {
        auto [pvov, pvom, pmov] = testing::make_pimpl<field_type>();
        vov_buffer_obt          = std::make_unique<buffer_type>(pvov->clone());
        vom_buffer_obt          = std::make_unique<buffer_type>(pvom->clone());
        mov_buffer_obt          = std::make_unique<buffer_type>(pmov->clone());

        // TODO: Test SET
        // ta_trange_type se_tr_vec{{0, 1, 2, 3}};
        // ta_trange_type se_tr_mat{{0, 1, 2}, {0, 1, 2}};
        // ta_trange_type se_tr_ten{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
        // pv->retile(se_tr_vec);
        // pm->retile(se_tr_mat);
        // pt->retile(se_tr_ten);
        // vec_buffer_set = std::make_unique<buffer_type>(std::move(pv));
        // mat_buffer_set = std::make_unique<buffer_type>(std::move(pm));
        // t3d_buffer_set = std::make_unique<buffer_type>(std::move(pt));
    }

    extents_type vector_extents{3}, matrix_extents{2, 2};
    auto vov_shape =
      testing::make_uniform_tot_shape(vector_extents, vector_extents);
    auto vom_shape =
      testing::make_uniform_tot_shape(vector_extents, matrix_extents);
    auto mov_shape =
      testing::make_uniform_tot_shape(matrix_extents, vector_extents);

    auto from_buffer = [](auto&& b) {
        return std::make_unique<buffer_type>(b->pimpl()->clone());
    };

    pimpl_type vov(from_buffer(vov_buffer_obt), vov_shape.clone(),
                   palloc->clone());
    pimpl_type vom(from_buffer(vom_buffer_obt), vom_shape.clone(),
                   palloc->clone());
    pimpl_type mov(from_buffer(mov_buffer_obt), mov_shape.clone(),
                   palloc->clone());

    // TODO: Test SET
    // pimpl_type vov2(from_buffer(vov_buffer_set), vov_shape->clone(),
    //             oalloc->clone());
    // pimpl_type vom2(from_buffer(vom_buffer_set), vom_shape->clone(),
    //             oalloc->clone());
    // pimpl_type mov2(from_buffer(mov_buffer_set), mov_shape->clone(),
    //             oalloc->clone());

    SECTION("CTors") {
        SECTION("From Components") {
            REQUIRE(vov.allocator() == *palloc);
            REQUIRE(vov.shape() == vov_shape);
            REQUIRE(vov.buffer() == *vov_buffer_obt);
            REQUIRE(vov.size() == 3);

            REQUIRE(vom.allocator() == *palloc);
            REQUIRE(vom.shape() == vom_shape);
            REQUIRE(vom.buffer() == *vom_buffer_obt);
            REQUIRE(vom.size() == 3);

            REQUIRE(mov.allocator() == *palloc);
            REQUIRE(mov.shape() == mov_shape);
            REQUIRE(mov.buffer() == *mov_buffer_obt);
            REQUIRE(mov.size() == 4);
        }

        SECTION("clone") {
            auto vov_copy = vov.clone();
            REQUIRE(*vov_copy == vov);
            // Make sure we didn't just alias
            REQUIRE(&vov_copy->allocator() != &vov.allocator());
            REQUIRE(&vov_copy->shape() != &vov.shape());

            REQUIRE(*(vom.clone()) == vom);
            REQUIRE(*(mov.clone()) == mov);
        }
    }

    SECTION("make_annotation") {
        REQUIRE(vov.make_annotation("i") == "i0;i1");
        REQUIRE(mov.make_annotation("j") == "j0,j1;j2");
        REQUIRE(vom.make_annotation("jk") == "jk0;jk1,jk2");
    }

    SECTION("rank") {
        REQUIRE(vov.rank() == 2);
        REQUIRE(mov.rank() == 3);
        REQUIRE(vom.rank() == 3);
    }

    SECTION("norm()") {
        REQUIRE(vov.norm() == Approx(6.4807406984).margin(1E-8));
        REQUIRE(mov.norm() == Approx(7.4833147735).margin(1E-8));
        REQUIRE(vom.norm() == Approx(9.4868329805).margin(1E-8));
    }

    SECTION("sum()") {
        REQUIRE(vov.sum() == 18);
        REQUIRE(mov.sum() == 24);
        REQUIRE(vom.sum() == 30);
    }

    SECTION("trace()") {
        REQUIRE_THROWS_AS(vov.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(mov.trace(), std::runtime_error);
        REQUIRE_THROWS_AS(vom.trace(), std::runtime_error);
    }

    SECTION("print") {
        std::stringstream ss;

        SECTION("vector-of-vectors") {
            auto pss = &(vov.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0], [3] ) {\n"
                               "  [0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [2]:[ [0], [3] ) { 1 2 3 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("matrix-of-vectors") {
            auto pss = &(mov.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0,0], [2,2] ) {\n"
                               "  [0,0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [0,1]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1,0]:[ [0], [3] ) { 1 2 3 }\n"
                               "  [1,1]:[ [0], [3] ) { 1 2 3 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }

        SECTION("vector-of-matrices") {
            auto pss = &(vom.print(ss));

            // Returns ss for chaining
            REQUIRE(pss == &ss);

            std::string corr = "0: [ [0], [3] ) {\n"
                               "  [0]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "  [1]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "  [2]:[ [0,0], [2,2] ) { 1 2 3 4 }\n"
                               "}\n";
            REQUIRE(corr == ss.str());
        }
    }

    SECTION("reallocate") {
        using except_t = std::runtime_error;

        SECTION("vector-of-vectors") {
            REQUIRE_THROWS_AS(vov.reallocate(oalloc->clone()), except_t);
        }

        SECTION("matrix-of-vectors") {
            REQUIRE_THROWS_AS(mov.reallocate(oalloc->clone()), except_t);
        }

        SECTION("vector-of-matrices") {
            REQUIRE_THROWS_AS(vom.reallocate(oalloc->clone()), except_t);
        }
    }

    SECTION("operator==") {
        SECTION("Same") {
            pimpl_type rhs(from_buffer(vom_buffer_obt), vom_shape.clone(),
                           palloc->clone());
            REQUIRE(vom == rhs);
        }

        SECTION("Different Values") {
            auto rhs_buffer = from_buffer(vom_buffer_obt);
            vom_buffer_obt->scale("i;j,k", "i;j,k", *rhs_buffer, 4.2);

            pimpl_type rhs(from_buffer(rhs_buffer), vom_shape.clone(),
                           palloc->clone());
            REQUIRE_FALSE(vom == rhs);
        }

        SECTION("Different shape") {
            using sparse_shape    = SparseShape<field_type>;
            using sparse_map_type = typename sparse_shape::sparse_map_type;
            using index_type      = typename sparse_map_type::key_type;

            index_type i0{0}, i1{1}, i2{2}, i00{0, 0}, i10{1, 0}, i01{0, 1},
              i11{1, 1};

            sparse_map_type sm{{i0, {i00, i01, i10, i11}},
                               {i1, {i00, i01, i10, i11}},
                               {i2, {i00, i01, i10, i11}}};
            auto new_shape = std::make_unique<sparse_shape>(
              vom.extents(), vom.shape().inner_extents(), sm);

            pimpl_type rhs(from_buffer(vom_buffer_obt), new_shape->clone(),
                           palloc->clone());

            REQUIRE_FALSE(vom == rhs);
        }
    }
}
