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
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper;
using namespace testing;
using namespace layout;

/* Testing Notes:
 *
 * - Much of the state of the Logical class is tested when testing the
 *   LayoutBase class. Here we focus on functionality defined/overridden in the
 *   Logical class.
 */
TEST_CASE("Logical") {
    shape::Smooth matrix_shape{2, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group no_symm, symm{p01};
    sparsity::Pattern no_sparsity;

    Logical matrix(matrix_shape, no_symm, no_sparsity);
    Logical symm_matrix(matrix_shape, symm, no_sparsity);

    SECTION("Ctors and assignment") {
        SECTION("Value") {
            REQUIRE(matrix.symmetry() == no_symm);
            REQUIRE(matrix.sparsity() == no_sparsity);

            REQUIRE(symm_matrix.symmetry() == symm);
            REQUIRE(symm_matrix.sparsity() == no_sparsity);
        }
    }

    SECTION("Virtual method overrides") {
        using const_base_reference = Logical::const_layout_reference;

        const_base_reference matrix_base      = matrix;
        const_base_reference symm_matrix_base = symm_matrix;

        SECTION("clone_") {
            REQUIRE(matrix_base.clone()->are_equal(matrix));
            REQUIRE(symm_matrix_base.clone()->are_equal(symm_matrix));
        }

        SECTION("are_equal") {
            REQUIRE(matrix_base.are_equal(matrix_base));
            REQUIRE_FALSE(symm_matrix_base.are_equal(matrix_base));
        }
    }
}
