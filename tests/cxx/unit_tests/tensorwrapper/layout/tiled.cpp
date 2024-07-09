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
#include <tensorwrapper/layout/mono_tile.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper;
using namespace testing;
using namespace layout;

/* Testing Notes:
 *
 * - Right now Tiled is an abstract class so we test methods implemented in it
 *   by creating MonoTile objects (which are not abstract).
 */

TEST_CASE("Tiled") {
    shape::Smooth matrix_shape{2, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group no_symm, symm{p01};
    sparsity::Pattern no_sparsity;

    MonoTile defaulted_mono;
    MonoTile matrix_mono(matrix_shape, no_symm, no_sparsity);
    MonoTile symm_matrix_mono(matrix_shape, symm, no_sparsity);

    // Test via references to the base class
    Tiled& defaulted   = defaulted_mono;
    Tiled& matrix      = matrix_mono;
    Tiled& symm_matrix = symm_matrix_mono;

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

    SECTION("has_shape") {
        REQUIRE_FALSE(defaulted.has_shape());
        REQUIRE(matrix.has_shape());
        REQUIRE(symm_matrix.has_shape());
    }

    SECTION("symmetry") {
        REQUIRE(defaulted.symmetry() == no_symm);
        REQUIRE(matrix.symmetry() == no_symm);
        REQUIRE(symm_matrix.symmetry() == symm);
    }

    SECTION("sparsity") {
        REQUIRE(defaulted.sparsity() == no_sparsity);
        REQUIRE(matrix.sparsity() == no_sparsity);
        REQUIRE(symm_matrix.sparsity() == no_sparsity);
    }

    SECTION("operator==") {
        // Defaulted v defaulted
        REQUIRE(defaulted == MonoTile{});

        // Different shape
        REQUIRE_FALSE(defaulted == matrix);

        // Different symmetry
        REQUIRE_FALSE(matrix == symm_matrix);
    }
}
