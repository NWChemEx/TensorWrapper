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

    Logical logi_copy_no_sym(matrix_shape, no_symm, no_sparsity);
    Logical logi_copy_has_sym(matrix_shape, symm, no_sparsity);
    Logical logi_copy_just_shape(matrix_shape);

    SECTION("Ctors and assignment") {
        SECTION("Value") {
            REQUIRE(logi_copy_no_sym.symmetry() == no_symm);
            REQUIRE(logi_copy_no_sym.sparsity() == no_sparsity);

            REQUIRE(logi_copy_has_sym.symmetry() == symm);
            REQUIRE(logi_copy_has_sym.sparsity() == no_sparsity);

            REQUIRE(logi_copy_just_shape.symmetry() == no_symm);
            REQUIRE(logi_copy_just_shape.sparsity() == no_sparsity);
        }
    }

    SECTION("Virtual method overrides") {
        using const_base_reference = Logical::const_layout_reference;

        const_base_reference base_copy_no_sym     = logi_copy_no_sym;
        const_base_reference base_copy_has_sym    = logi_copy_has_sym;
        const_base_reference base_copy_just_shape = logi_copy_just_shape;

        SECTION("clone_") {
            REQUIRE(base_copy_no_sym.clone()->are_equal(logi_copy_no_sym));
            REQUIRE(base_copy_has_sym.clone()->are_equal(logi_copy_has_sym));
            REQUIRE(
              base_copy_just_shape.clone()->are_equal(logi_copy_just_shape));
        }

        SECTION("are_equal") {
            REQUIRE(base_copy_no_sym.are_equal(base_copy_no_sym));
            REQUIRE_FALSE(base_copy_has_sym.are_equal(base_copy_no_sym));
            REQUIRE(base_copy_just_shape.are_equal(base_copy_no_sym));
        }
    }
}
