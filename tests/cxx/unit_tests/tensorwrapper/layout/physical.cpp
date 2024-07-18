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
 * - Much of the state of the Physical class is tested when testing the
 *   LayoutBase class. Here we focus on functionality defined/overridden in the
 *   Physical class.
 */
TEST_CASE("Physical") {
    shape::Smooth matrix_shape{2, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group no_symm, symm{p01};
    sparsity::Pattern no_sparsity;

    Physical defaulted;
    Physical matrix(matrix_shape, no_symm, no_sparsity);
    Physical symm_matrix(matrix_shape, symm, no_sparsity);

    SECTION("Ctors and assignment") {
        SECTION("Defaulted") {
            REQUIRE_FALSE(defaulted.has_shape());
            REQUIRE(defaulted.symmetry() == no_symm);
            REQUIRE(defaulted.sparsity() == no_sparsity);
        }

        SECTION("Value") {
            REQUIRE(matrix.has_shape());
            REQUIRE(matrix.symmetry() == no_symm);
            REQUIRE(matrix.sparsity() == no_sparsity);

            REQUIRE(symm_matrix.has_shape());
            REQUIRE(symm_matrix.symmetry() == symm);
            REQUIRE(symm_matrix.sparsity() == no_sparsity);
        }
    }

    SECTION("Virtual method overrides") {
        using const_base_reference = Physical::const_layout_reference;

        const_base_reference defaulted_base   = defaulted;
        const_base_reference matrix_base      = matrix;
        const_base_reference symm_matrix_base = symm_matrix;

        SECTION("clone_") {
            REQUIRE(defaulted_base.clone()->are_equal(defaulted));
            REQUIRE(matrix_base.clone()->are_equal(matrix));
            REQUIRE(symm_matrix_base.clone()->are_equal(symm_matrix));
        }

        SECTION("are_equal") {
            REQUIRE(defaulted_base.are_equal(defaulted_base));
            REQUIRE_FALSE(defaulted_base.are_equal(matrix_base));
        }
    }
}
