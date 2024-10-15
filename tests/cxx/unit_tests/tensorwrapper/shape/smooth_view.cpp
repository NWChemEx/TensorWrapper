/*
 * Copyright 2024 NWChemEx Community
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
#include <set>
#include <tensorwrapper/shape/smooth_view.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::shape;

using rank_type = typename Smooth::rank_type;
using size_type = typename Smooth::size_type;

using types2test = std::pair<Smooth, const Smooth>;
TEMPLATE_LIST_TEST_CASE("SmoothView", "", types2test) {
    using view_type       = SmoothView<TestType>;
    using const_view_type = SmoothView<const std::decay_t<TestType>>;
    using smooth_type     = typename view_type::smooth_type;

    smooth_type scalar{};
    smooth_type vector{3};

    view_type alias_scalar(scalar);
    view_type alias_vector(vector);

    SECTION("Ctors and assignment") {
        SECTION("alias Smooth object") {
            REQUIRE(alias_scalar.rank() == rank_type(0));
            REQUIRE(alias_scalar.size() == size_type(1));

            REQUIRE(alias_vector.rank() == rank_type(1));
            REQUIRE(alias_vector.size() == size_type(3));
        }

        SECTION("Assign to const") {
            // if TestType is non-const this tests mutable to const conversion,
            // otherwise this duplicates the copy ctor test.
            const_view_type const_scalar(alias_scalar);
            REQUIRE(const_scalar.rank() == rank_type(0));
            REQUIRE(const_scalar.size() == size_type(1));
        }

        test_copy_and_move_ctors(alias_scalar, alias_vector);

        SECTION("copy assignment") {
            view_type copy_scalar(alias_scalar);
            auto pcopy_scalar = &(copy_scalar = alias_vector);
            REQUIRE(copy_scalar == alias_vector);
            REQUIRE(pcopy_scalar == &copy_scalar);
        }

        SECTION("move assignment") {
            view_type copy_scalar(alias_scalar);
            view_type copy_vector(alias_vector);
            auto pcopy_scalar = &(copy_scalar = std::move(alias_vector));
            REQUIRE(copy_scalar == copy_vector);
            REQUIRE(pcopy_scalar == &copy_scalar);
        }
    }

    SECTION("extent") {
        REQUIRE_THROWS_AS(alias_scalar.extent(0), std::out_of_range);

        REQUIRE(alias_vector.extent(0) == 3);
        REQUIRE_THROWS_AS(alias_vector.extent(1), std::out_of_range);
    }

    SECTION("rank") {
        REQUIRE(alias_scalar.rank() == rank_type(0));
        REQUIRE(alias_vector.rank() == rank_type(1));
    }

    SECTION("size") {
        REQUIRE(alias_scalar.size() == size_type(1));
        REQUIRE(alias_vector.size() == size_type(3));
    }

    SECTION("Utility methods") {
        SECTION("swap") {
            view_type scalar_copy(alias_scalar);
            view_type vector_copy(alias_vector);

            alias_vector.swap(alias_scalar);
            REQUIRE(alias_vector == scalar_copy);
            REQUIRE(alias_scalar == vector_copy);
        }

        SECTION("operator==") {
            // Same shapes
            REQUIRE(alias_scalar == view_type(scalar));
            REQUIRE(alias_vector == view_type(vector));

            // (Possibly) different const-ness (if same const-ness duplicates
            // the above check). Also check for symmetry.
            REQUIRE(alias_scalar == const_view_type(alias_scalar));
            REQUIRE(const_view_type(alias_scalar) == alias_scalar);

            // Can compare aliases with objects
            REQUIRE(alias_scalar == scalar);

            // Different ranks
            REQUIRE_FALSE(alias_scalar == alias_vector);

            // Different extents
            smooth_type vector2{2};
            REQUIRE_FALSE(alias_vector == view_type(vector2));
        }

        SECTION("operator!=") {
            // Implemented by negating operator==, so just spot check
            REQUIRE_FALSE(alias_scalar != view_type(scalar));
            REQUIRE(alias_scalar != alias_vector);
        }
    }
}
