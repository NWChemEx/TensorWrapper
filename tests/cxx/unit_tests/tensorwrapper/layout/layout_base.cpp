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
#include "../helpers.hpp"
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper;
using namespace testing;
using namespace layout;

/* Testing Notes:
 *
 * - Right now LayoutBase is an abstract class so we test methods implemented in
 *   it by creating Physical objects (which are not abstract).
 */

TEST_CASE("LayoutBase") {
    shape::Smooth matrix_shape{2, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group no_symm, symm{p01};
    sparsity::Pattern no_sparsity;

    Physical phys_copy_no_sym(matrix_shape, no_symm, no_sparsity);
    Physical phys_copy_has_sym(matrix_shape, symm, no_sparsity);
    Physical phys_copy_just_shape(matrix_shape);

    // Test via references to the base class
    LayoutBase& base_copy_no_sym     = phys_copy_no_sym;
    LayoutBase& base_copy_has_sym    = phys_copy_has_sym;
    LayoutBase& base_copy_just_shape = phys_copy_just_shape;

    SECTION("Ctors and assignment") {
        SECTION("Copy state") {
            REQUIRE(base_copy_no_sym.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_no_sym.symmetry() == no_symm);
            REQUIRE(base_copy_no_sym.sparsity() == no_sparsity);

            REQUIRE(base_copy_has_sym.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_has_sym.symmetry() == symm);
            REQUIRE(base_copy_has_sym.sparsity() == no_sparsity);

            REQUIRE(base_copy_just_shape.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_just_shape.symmetry() == no_symm);
            REQUIRE(base_copy_just_shape.sparsity() == no_sparsity);
        }

        SECTION("Move state") {
            auto pshape  = matrix_shape.clone();
            auto psymm   = std::make_unique<symmetry::Group>(no_symm);
            auto psparse = std::make_unique<sparsity::Pattern>(no_sparsity);
            SECTION("All non-null") {
                Physical rhs(std::move(pshape), std::move(psymm),
                             std::move(psparse));
                REQUIRE(base_copy_no_sym == rhs);
            }
            SECTION("Only Shape input") {
                Physical rhs(std::move(pshape));
                REQUIRE(base_copy_just_shape == rhs);
            }
            SECTION("Shape is null") {
                REQUIRE_THROWS_AS(
                  Physical(nullptr, std::move(psymm), std::move(psparse)),
                  std::runtime_error);
                REQUIRE_THROWS_AS(Physical(nullptr), std::runtime_error);
            }
            SECTION("Symmetry is null") {
                REQUIRE_THROWS_AS(
                  Physical(std::move(pshape), nullptr, std::move(psparse)),
                  std::runtime_error);
            }
            SECTION("Sparsity is null") {
                REQUIRE_THROWS_AS(
                  Physical(std::move(pshape), std::move(psymm), nullptr),
                  std::runtime_error);
            }
        }
    }

    SECTION("shape") {
        REQUIRE(base_copy_no_sym.shape().are_equal(matrix_shape));
        REQUIRE(base_copy_has_sym.shape().are_equal(matrix_shape));
    }

    SECTION("symmetry") {
        REQUIRE(base_copy_no_sym.symmetry() == no_symm);
        REQUIRE(base_copy_has_sym.symmetry() == symm);
    }

    SECTION("sparsity") {
        REQUIRE(base_copy_no_sym.sparsity() == no_sparsity);
        REQUIRE(base_copy_has_sym.sparsity() == no_sparsity);
    }

    SECTION("operator==") {
        // Same
        REQUIRE(base_copy_no_sym ==
                Physical(matrix_shape, no_symm, no_sparsity));

        // Different shape
        shape::Smooth vector_shape{2};
        REQUIRE_FALSE(base_copy_no_sym ==
                      Physical(vector_shape, no_symm, no_sparsity));

        // Different symmetry
        REQUIRE_FALSE(base_copy_no_sym == base_copy_has_sym);
    }
}
