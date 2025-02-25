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

using namespace tensorwrapper;

/* Testing strategy:
 *
 * There's a plethora of possible states a ModuleInput object can be in. We're
 * not going to test them all. Here we focus on testing ModuleInput objects
 * with states we expect to see. The TensorFactory class is ultimately
 * responsible for determining whether a particular ModuleInput state is valid
 * or not (as only it knows what default values it can compute from a set of
 * user-provided inputs).
 */

TEST_CASE("TensorInput") {
    // We just test with some stub objects
    parallelzone::runtime::RuntimeView rv;
    shape::Smooth shape{3, 3};
    symmetry::Group g{symmetry::Permutation{0, 1}};
    sparsity::Pattern sparsity(2);
    layout::Logical logical(shape, g, sparsity);
    layout::Physical physical(shape, g, sparsity);
    allocator::Eigen<double> alloc(rv);
    auto pbuffer = alloc.construct(42.0);
    auto& buffer = *pbuffer;

    detail_::TensorInput defaulted;
    detail_::TensorInput scalar(shape::Smooth{});
    detail_::TensorInput symm_matrix(shape, g);

    SECTION("Ctors") {
        SECTION("Default") {
            REQUIRE(defaulted.m_pshape == nullptr);
            REQUIRE(defaulted.m_psymmetry == nullptr);
            REQUIRE(defaulted.m_psparsity == nullptr);
            REQUIRE(defaulted.m_plogical == nullptr);
            REQUIRE(defaulted.m_pphysical == nullptr);
            REQUIRE(defaulted.m_palloc == nullptr);
            REQUIRE(defaulted.m_pbuffer == nullptr);
            REQUIRE(defaulted.m_rv == rv);
        }

        SECTION("Shape (by value)") {
            REQUIRE(scalar.m_pshape->are_equal(shape::Smooth{}));
            REQUIRE(scalar.m_psymmetry == nullptr);
            REQUIRE(scalar.m_psparsity == nullptr);
            REQUIRE(scalar.m_plogical == nullptr);
            REQUIRE(scalar.m_pphysical == nullptr);
            REQUIRE(scalar.m_palloc == nullptr);
            REQUIRE(scalar.m_pbuffer == nullptr);
            REQUIRE(scalar.m_rv == rv);

            REQUIRE(scalar.has_shape());
        }

        SECTION("Shape (by pointer)") {
            shape::Smooth scalar_shape{};
            auto pscalar_shape        = scalar_shape.clone();
            auto scalar_shape_address = pscalar_shape.get();
            detail_::TensorInput i(std::move(pscalar_shape));
            REQUIRE(scalar_shape.are_equal(*i.m_pshape));
            REQUIRE(i.m_pshape.get() == scalar_shape_address);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(scalar.has_shape());
        }

        SECTION("Group (by value)") {
            REQUIRE(symm_matrix.m_pshape->are_equal(shape));
            REQUIRE(*symm_matrix.m_psymmetry == g);
            REQUIRE(symm_matrix.m_psparsity == nullptr);
            REQUIRE(symm_matrix.m_plogical == nullptr);
            REQUIRE(symm_matrix.m_pphysical == nullptr);
            REQUIRE(symm_matrix.m_palloc == nullptr);
            REQUIRE(symm_matrix.m_pbuffer == nullptr);
            REQUIRE(symm_matrix.m_rv == rv);

            REQUIRE(symm_matrix.has_symmetry());
        }

        SECTION("Group (by pointer)") {
            auto pg         = std::make_unique<symmetry::Group>(g);
            auto pg_address = pg.get();
            detail_::TensorInput i(std::move(pg), shape);
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(i.m_psymmetry.get() == pg_address);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_symmetry());
        }

        SECTION("Sparsity (by value)") {
            detail_::TensorInput i(g, shape, sparsity);
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(*i.m_psparsity == sparsity);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_sparsity());
        }

        SECTION("Sparsity (by pointer)") {
            auto psparsity = std::make_unique<sparsity::Pattern>(sparsity);
            auto psparsity_address = psparsity.get();
            detail_::TensorInput i(g, shape, std::move(psparsity));
            REQUIRE(i.m_pshape->are_equal(shape));
            REQUIRE(*i.m_psymmetry == g);
            REQUIRE(*i.m_psparsity == sparsity);
            REQUIRE(i.m_psparsity.get() == psparsity_address);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_sparsity());
        }

        SECTION("Logical (by value)") {
            detail_::TensorInput i(logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_logical_layout());
        }

        SECTION("Logical (by pointer)") {
            auto plogical         = std::make_unique<layout::Logical>(logical);
            auto plogical_address = plogical.get();
            detail_::TensorInput i(std::move(plogical));
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_plogical.get() == plogical_address);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_logical_layout());
        }

        SECTION("Physical (by value)") {
            detail_::TensorInput i(physical, logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_physical_layout());
        }

        SECTION("Physical (by pointer)") {
            auto pphysical = std::make_unique<layout::Physical>(physical);
            auto pphysical_address = pphysical.get();
            detail_::TensorInput i(std::move(pphysical), logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_pphysical.get() == pphysical_address);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_physical_layout());
        }

        SECTION("Allocator (by value)") {
            detail_::TensorInput i(physical, alloc, logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc->are_equal(alloc));
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_allocator());
        }

        SECTION("Allocator (by pointer)") {
            auto palloc        = alloc.clone();
            auto alloc_address = palloc.get();
            detail_::TensorInput i(physical, std::move(palloc), logical);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc->are_equal(alloc));
            REQUIRE(i.m_palloc.get() == alloc_address);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_allocator());
        }

        SECTION("Buffer (by value)") {
            detail_::TensorInput i(physical, alloc, logical, buffer);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc->are_equal(alloc));
            // REQUIRE(i.m_pbuffer->are_equal(buffer));
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_buffer());
        }

        SECTION("Buffer (by pointer)") {
            auto pbuffer        = buffer.clone();
            auto buffer_address = pbuffer.get();
            detail_::TensorInput i(physical, alloc, logical,
                                   std::move(pbuffer));
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical->are_equal(logical));
            REQUIRE(i.m_pphysical->are_equal(physical));
            REQUIRE(i.m_palloc->are_equal(alloc));
            // REQUIRE(i.m_pbuffer->are_equal(buffer));
            REQUIRE(i.m_pbuffer.get() == buffer_address);
            REQUIRE(i.m_rv == rv);

            REQUIRE(i.has_buffer());
        }

        SECTION("RuntimeView") {
            detail_::TensorInput i(rv);
            REQUIRE(i.m_pshape == nullptr);
            REQUIRE(i.m_psymmetry == nullptr);
            REQUIRE(i.m_psparsity == nullptr);
            REQUIRE(i.m_plogical == nullptr);
            REQUIRE(i.m_pphysical == nullptr);
            REQUIRE(i.m_palloc == nullptr);
            REQUIRE(i.m_pbuffer == nullptr);
            REQUIRE(i.m_rv == rv);
        }
    }

    SECTION("has_shape") {
        REQUIRE_FALSE(defaulted.has_shape());
        REQUIRE(scalar.has_shape());
    }

    SECTION("has_symmetry") {
        REQUIRE_FALSE(defaulted.has_symmetry());
        REQUIRE(symm_matrix.has_symmetry());
    }

    SECTION("has_sparsity") {
        REQUIRE_FALSE(defaulted.has_sparsity());

        detail_::TensorInput w_sparsity(sparsity);
        REQUIRE(w_sparsity.has_sparsity());
    }

    SECTION("has_logical_layout") {
        REQUIRE_FALSE(defaulted.has_logical_layout());

        detail_::TensorInput w_logical(logical);
        REQUIRE(w_logical.has_logical_layout());
    }

    SECTION("has_physical_layout") {
        REQUIRE_FALSE(defaulted.has_physical_layout());

        detail_::TensorInput w_physical(physical);
        REQUIRE(w_physical.has_physical_layout());
    }

    SECTION("has_allocator") {
        REQUIRE_FALSE(defaulted.has_allocator());

        detail_::TensorInput w_allocator(alloc);
        REQUIRE(w_allocator.has_allocator());
    }

    SECTION("has_buffer") {
        REQUIRE_FALSE(defaulted.has_buffer());

        detail_::TensorInput w_buffer(buffer);
        REQUIRE(w_buffer.has_buffer());
    }
}
