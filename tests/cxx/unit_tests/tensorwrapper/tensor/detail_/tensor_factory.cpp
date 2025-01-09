/*
 * Copyright 2024 NWChemEx-Project
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

#include "../../testing/testing.hpp"
#include <tensorwrapper/tensor/detail_/tensor_factory.hpp>
#include <tensorwrapper/tensor/detail_/tensor_pimpl.hpp>

using namespace tensorwrapper;

/* Testing strategy:
 *
 * TensorInput already had too many input states to test. TensorFactory has even
 * more because
 */

TEST_CASE("TensorFactory") {
    using detail_::TensorFactory;
    using detail_::TensorInput;

    parallelzone::runtime::RuntimeView rv;

    shape::Smooth shape{};
    auto pshape        = shape.clone();
    auto shape_address = pshape.get();

    symmetry::Group g;
    auto pg        = std::make_unique<symmetry::Group>(g);
    auto g_address = pg.get();

    sparsity::Pattern sparsity;
    auto psparsity        = std::make_unique<sparsity::Pattern>(sparsity);
    auto sparsity_address = psparsity.get();

    layout::Logical logical(shape, g, sparsity);
    auto plogical        = logical.clone_as<layout::Logical>();
    auto logical_address = plogical.get();

    layout::Physical physical(shape, g, sparsity);
    auto pphysical = physical.clone_as<layout::Physical>();

    allocator::Eigen<double, 0> alloc(rv);
    auto pbuffer        = alloc.allocate(std::move(pphysical));
    auto buffer_address = pbuffer.get();

    SECTION("default_logical_symmetry") {
        // N.B. at moment default symmetry is no-symmetry, i.e., an empty Group
        symmetry::Group corr;
        auto i      = testing::smooth_scalar_input();
        auto result = TensorFactory::default_logical_symmetry(*i.m_pshape);
        REQUIRE((*result) == corr);
    }

    SECTION("default_logical_sparsity") {
        // N.B. at moment default symmetry is no sparsity
        sparsity::Pattern corr(2);

        auto i = testing::smooth_symmetric_matrix_input();
        auto result =
          TensorFactory::default_logical_sparsity(*i.m_pshape, *i.m_psymmetry);
        REQUIRE((*result) == corr);
    }

    SECTION("default_logical_layout") {
        // N.B. at moment just wraps the shape, symmetry, and sparsity provided
        auto result = TensorFactory::default_logical_layout(
          std::move(pshape), std::move(pg), std::move(psparsity));
        REQUIRE(result->are_equal(logical));
        REQUIRE(&result->shape() == shape_address);
        REQUIRE(&result->symmetry() == g_address);
        REQUIRE(&result->sparsity() == sparsity_address);
    }

    SECTION("default_physical_layout") {
        auto result = TensorFactory::default_physical_layout(logical);
        REQUIRE(result->are_equal(physical));
    }

    SECTION("default_allocator") {
        auto result = TensorFactory::default_allocator(physical, rv);
        REQUIRE(result->are_equal(alloc));
    }

    SECTION("construct(input)") {
        SECTION("Can create default pimpl") {
            auto pdefaulted = TensorFactory::construct(TensorInput{});
            REQUIRE(pdefaulted == nullptr);
        }

        SECTION("Logical layout & Buffer") {
            TensorInput i(std::move(plogical), std::move(pbuffer));
            auto ppimpl = TensorFactory::construct(std::move(i));
            REQUIRE(&ppimpl->logical_layout() == logical_address);
            REQUIRE(&ppimpl->buffer() == buffer_address);
        }

        SECTION("Throws if invalid") {
            TensorInput i(std::move(pbuffer));
            using except_t = std::runtime_error;
            REQUIRE_THROWS_AS(TensorFactory::construct(std::move(i)), except_t);
        }
    }

    SECTION("construct(scalar_il_type)") {
        auto ppimpl = TensorFactory::construct(42.0);
        auto corr   = TensorFactory::construct(testing::smooth_scalar_input());
        REQUIRE(*ppimpl == *corr);
    }

    SECTION("construct(vector_il_type)") {
        using vector_il_type = typename TensorFactory::vector_il_type;
        vector_il_type il{0.0, 1.0, 2.0, 3.0, 4.0};
        auto ppimpl = TensorFactory::construct(il);
        auto corr   = TensorFactory::construct(testing::smooth_vector_input());
        REQUIRE(*ppimpl == *corr);
    }

    SECTION("construct(matrix_il_type)") {
        using matrix_il_type = typename TensorFactory::matrix_il_type;
        matrix_il_type il{{1.0, 2.0}, {3.0, 4.0}};
        auto ppimpl = TensorFactory::construct(il);
        auto corr   = TensorFactory::construct(testing::smooth_matrix_input());
        REQUIRE(*ppimpl == *corr);
    }

    SECTION("construct(tensor3_il_type)") {
        using tensor3_il_type = typename TensorFactory::tensor3_il_type;
        tensor3_il_type il{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
        auto ppimpl = TensorFactory::construct(il);
        auto corr   = TensorFactory::construct(testing::smooth_tensor3_input());
        REQUIRE(*ppimpl == *corr);
    }

    SECTION("construct(tensor4_il_type)") {
        using tensor4_il_type = typename TensorFactory::tensor4_il_type;
        tensor4_il_type il{
          {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
          {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}};
        auto ppimpl = TensorFactory::construct(il);
        auto corr   = TensorFactory::construct(testing::smooth_tensor4_input());
        REQUIRE(*ppimpl == *corr);
    }

    SECTION("can_make_logical_layout") {
        TensorFactory f;
        REQUIRE(f.can_make_logical_layout(TensorInput(shape)));
        REQUIRE(f.can_make_logical_layout(TensorInput(logical)));
        REQUIRE_FALSE(f.can_make_logical_layout(TensorInput{}));
        REQUIRE_FALSE(f.can_make_logical_layout(TensorInput(physical)));
    }

    SECTION("assert_valid") {
        TensorFactory f;
        REQUIRE_NOTHROW(f.assert_valid(testing::smooth_scalar_input()));
        REQUIRE_NOTHROW(f.assert_valid(testing::smooth_vector_input()));
        REQUIRE_NOTHROW(
          f.assert_valid(testing::smooth_symmetric_matrix_input()));

        using e_t = std::runtime_error;

        SECTION("Layout with incompatible shape") {
            TensorInput i(shape::Smooth{3, 3}, logical);
            REQUIRE_THROWS_AS(f.assert_valid(i), e_t);
        }

        SECTION("Layout with incompatible symmetry") {
            symmetry::Group g0{symmetry::Permutation{0, 1}};
            TensorInput i(g0, logical);
            REQUIRE_THROWS_AS(f.assert_valid(i), e_t);
        }

        SECTION("Buffer with incompatible physical layout") {
            layout::Physical p(shape::Smooth{3, 3});
            TensorInput i(std::move(pbuffer), p);
            REQUIRE_THROWS_AS(f.assert_valid(i), e_t);
        }

        SECTION("only buffer") {
            TensorInput i(std::move(pbuffer));
            REQUIRE_THROWS_AS(f.assert_valid(i), e_t);
        }

        SECTION("only physical layout") {
            TensorInput i(physical);
            REQUIRE(i.has_physical_layout());
            REQUIRE_THROWS_AS(f.assert_valid(i), e_t);
        }

        SECTION("logical layout and buffer (should work)") {
            TensorInput i(std::move(logical), std::move(pbuffer));
            REQUIRE(i.has_logical_layout());
            REQUIRE(i.has_buffer());
            REQUIRE_NOTHROW(f.assert_valid(i));
        }
    }
}
