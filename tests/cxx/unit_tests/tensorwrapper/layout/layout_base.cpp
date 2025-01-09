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
#include "../testing/testing.hpp"
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
    symmetry::Permutation p01{1, 0};
    symmetry::Group no_symm(2), symm{p01};
    sparsity::Pattern no_sparsity(2);

    Physical phys_copy_no_sym(matrix_shape, no_symm, no_sparsity);
    Physical phys_copy_has_sym(matrix_shape, symm, no_sparsity);
    Physical phys_copy_just_shape(matrix_shape);

    // Test via references to the base class
    LayoutBase& base_copy_no_sym     = phys_copy_no_sym;
    LayoutBase& base_copy_has_sym    = phys_copy_has_sym;
    LayoutBase& base_copy_just_shape = phys_copy_just_shape;

    SECTION("Ctors") {
        SECTION("Copy state") {
            REQUIRE(base_copy_no_sym.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_no_sym.symmetry().are_equal(no_symm));
            REQUIRE(base_copy_no_sym.sparsity().are_equal(no_sparsity));

            REQUIRE(base_copy_has_sym.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_has_sym.symmetry().are_equal(symm));
            REQUIRE(base_copy_has_sym.sparsity().are_equal(no_sparsity));

            REQUIRE(base_copy_just_shape.shape().are_equal(matrix_shape));
            REQUIRE(base_copy_just_shape.symmetry().are_equal(no_symm));
            REQUIRE(base_copy_just_shape.sparsity().are_equal(no_sparsity));
        }

        SECTION("Copy shape, default others") {
            Physical only_shape(matrix_shape);
            REQUIRE(only_shape.shape().are_equal(matrix_shape));
            REQUIRE(only_shape.symmetry().are_equal(no_symm));
            REQUIRE(only_shape.sparsity().are_equal(no_sparsity));
        }

        SECTION("Move shape, default others") {
            Physical only_shape(matrix_shape.clone());
            REQUIRE(only_shape.shape().are_equal(matrix_shape));
            REQUIRE(only_shape.symmetry().are_equal(no_symm));
            REQUIRE(only_shape.sparsity().are_equal(no_sparsity));
        }

        SECTION("Move state") {
            auto pshape  = matrix_shape.clone();
            auto psymm   = no_symm.clone();
            auto psparse = no_sparsity.clone();
            SECTION("All non-null") {
                Physical rhs(std::move(pshape), std::move(psymm),
                             std::move(psparse));
                REQUIRE(base_copy_no_sym == rhs);
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
            SECTION("Shape ranks is inconsistent") {
                shape::Smooth s0{};
                REQUIRE_THROWS_AS(
                  Physical(s0.clone(), std::move(psymm), std::move(psparse)),
                  std::runtime_error);
            }
            SECTION("Symmetry rank is inconsistent") {
                symmetry::Group g0(0);
                REQUIRE_THROWS_AS(
                  Physical(std::move(pshape), g0.clone(), std::move(psparse)),
                  std::runtime_error);
            }

            SECTION("Sparsity rank is inconsistent") {
                sparsity::Pattern p3(3);
                REQUIRE_THROWS_AS(
                  Physical(std::move(pshape), std::move(psymm), p3.clone()),
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

    SECTION("rank") {
        REQUIRE(base_copy_no_sym.rank() == 2);
        REQUIRE(base_copy_has_sym.rank() == 2);
    }

    SECTION("operator==") {
        // Same
        REQUIRE(base_copy_no_sym ==
                Physical(matrix_shape, no_symm, no_sparsity));

        // Different shape
        shape::Smooth matrix_shape2{3, 2};
        REQUIRE_FALSE(base_copy_no_sym ==
                      Physical(matrix_shape2, no_symm, no_sparsity));

        // Different symmetry
        REQUIRE_FALSE(base_copy_no_sym == base_copy_has_sym);

        // N.b. presently not possible to have different sparsities w/o
        // different ranks
    }

    SECTION("operator!=") {
        REQUIRE_FALSE(base_copy_no_sym !=
                      Physical(matrix_shape, no_symm, no_sparsity));
        REQUIRE(base_copy_no_sym != base_copy_has_sym);
    }

    SECTION("xxx_assignment_") {
        // Layout just calls the equivlanent xxx_assignment_ method on its
        // shape, symmetry, and sparsity objects. Spot checking works here if
        // since the called methods are tested
        auto scalar_layout = testing::scalar_physical();
        auto matrix_layout = testing::matrix_physical();

        SECTION("addition_assignment_") {
            auto lij  = matrix_layout("i,j");
            auto pout = &(scalar_layout.addition_assignment("i,j", lij, lij));
            REQUIRE(pout == &scalar_layout);
            REQUIRE(scalar_layout == matrix_layout);
        }

        SECTION("subtraction_assignment_") {
            auto lij = matrix_layout("i,j");
            auto pout =
              &(scalar_layout.subtraction_assignment("i,j", lij, lij));
            REQUIRE(pout == &scalar_layout);
            REQUIRE(scalar_layout == matrix_layout);
        }

        SECTION("multiplication_assignment_") {
            auto lij = matrix_layout("i,j");
            auto pout =
              &(scalar_layout.multiplication_assignment("i,j", lij, lij));
            REQUIRE(pout == &scalar_layout);
            REQUIRE(scalar_layout == matrix_layout);
        }

        SECTION("permute_assignment_") {
            auto lij  = matrix_layout("i,j");
            auto pout = &(scalar_layout.permute_assignment("i,j", lij));
            REQUIRE(pout == &scalar_layout);
            REQUIRE(scalar_layout == matrix_layout);
        }
    }
}
