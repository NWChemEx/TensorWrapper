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
 * - Much of the state of the MonoTile class is tested when testing the Tiled
 *   class. Here we focus on functionality defined/overridden in the MonoTile
 *   class.
 */
TEST_CASE("MonoTile") {
    shape::Smooth matrix_shape{2, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group no_symm, symm{p01};
    sparsity::Pattern no_sparsity;

    MonoTile defaulted;
    MonoTile matrix(matrix_shape, no_symm, no_sparsity);
    MonoTile symm_matrix(matrix_shape, symm, no_sparsity);

    SECTION("Ctors and assignment") {
        SECTION("Defaulted") {
            REQUIRE(defaulted.tile_size() == 0);
            REQUIRE_FALSE(defaulted.has_shape());
            REQUIRE(defaulted.symmetry() == no_symm);
            REQUIRE(defaulted.sparsity() == no_sparsity);
        }

        SECTION("Value") {
            REQUIRE(matrix.tile_size() == 1);
            REQUIRE(matrix.has_shape());
            REQUIRE(matrix.symmetry() == no_symm);
            REQUIRE(matrix.sparsity() == no_sparsity);

            REQUIRE(symm_matrix.tile_size() == 1);
            REQUIRE(symm_matrix.has_shape());
            REQUIRE(symm_matrix.symmetry() == symm);
            REQUIRE(symm_matrix.sparsity() == no_sparsity);
        }
    }

    SECTION("Virtual method overrides") {
        using const_base_reference = MonoTile::const_layout_reference;

        const_base_reference defaulted_base   = defaulted;
        const_base_reference matrix_base      = matrix;
        const_base_reference symm_matrix_base = symm_matrix;

        SECTION("clone_") {
            REQUIRE(defaulted_base.clone()->are_equal(defaulted));
            REQUIRE(matrix_base.clone()->are_equal(matrix));
            REQUIRE(symm_matrix_base.clone()->are_equal(symm_matrix));
        }

        SECTION("tile_size_") {
            REQUIRE(defaulted_base.tile_size() == 0);
            REQUIRE(matrix_base.tile_size() == 1);
            REQUIRE(symm_matrix_base.tile_size() == 1);
        }

        SECTION("are_equal") {
            REQUIRE(defaulted_base.are_equal(defaulted_base));
            REQUIRE_FALSE(defaulted_base.are_equal(matrix_base));
        }
    }
}
