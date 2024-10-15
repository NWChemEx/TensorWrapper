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
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::shape;

using rank_type = typename Smooth::rank_type;
using size_type = typename Smooth::size_type;

TEST_CASE("Smooth") {
    // Tests the initializer list ctor
    Smooth scalar{};
    Smooth vector{1};

    // Tests the range ctor with two different types of iterators
    std::vector<size_type> matrix_extents{2, 3};
    std::set<size_type> tensor_extents{3, 4, 5};

    Smooth matrix(matrix_extents.begin(), matrix_extents.end());
    Smooth tensor(tensor_extents.begin(), tensor_extents.end());

    SECTION("Ctors and assignment") {
        SECTION("Initializer list") {
            REQUIRE(scalar.rank() == rank_type(0));
            REQUIRE(scalar.size() == size_type(1));

            REQUIRE(vector.rank() == rank_type(1));
            REQUIRE(vector.size() == size_type(1));
        }

        SECTION("Range ctor") {
            REQUIRE(matrix.rank() == rank_type(2));
            REQUIRE(matrix.size() == size_type(6));

            REQUIRE(tensor.rank() == rank_type(3));
            REQUIRE(tensor.size() == size_type(60));
        }

        test_copy_move_ctor_and_assignment(scalar, vector, matrix, tensor);
    }

    SECTION("extent") {
        REQUIRE_THROWS_AS(scalar.extent(0), std::out_of_range);

        REQUIRE(vector.extent(0) == 1);
        REQUIRE_THROWS_AS(vector.extent(1), std::out_of_range);

        REQUIRE(matrix.extent(0) == matrix_extents[0]);
        REQUIRE(matrix.extent(1) == matrix_extents[1]);
        REQUIRE_THROWS_AS(matrix.extent(2), std::out_of_range);

        REQUIRE(tensor.extent(0) == 3);
        REQUIRE(tensor.extent(1) == 4);
        REQUIRE(tensor.extent(2) == 5);
        REQUIRE_THROWS_AS(tensor.extent(3), std::out_of_range);
    }

    SECTION("Virtual implementations") {
        SECTION("clone") {
            REQUIRE(scalar.clone()->are_equal(scalar));
            REQUIRE(vector.clone()->are_equal(vector));
            REQUIRE(matrix.clone()->are_equal(matrix));
            REQUIRE(tensor.clone()->are_equal(tensor));
        }

        SECTION("rank") {
            REQUIRE(scalar.rank() == rank_type(0));
            REQUIRE(vector.rank() == rank_type(1));
            REQUIRE(matrix.rank() == rank_type(2));
            REQUIRE(tensor.rank() == rank_type(3));
        }

        SECTION("size") {
            REQUIRE(scalar.size() == size_type(1));
            REQUIRE(vector.size() == size_type(1));
            REQUIRE(matrix.size() == size_type(6));
            REQUIRE(tensor.size() == size_type(60));
        }

        SECTION("as_smooth()") {
            REQUIRE(scalar.as_smooth() == scalar);
            REQUIRE(vector.as_smooth() == vector);
        }

        SECTION("as_smooth() const") {
            REQUIRE(std::as_const(scalar).as_smooth() == scalar);
            REQUIRE(std::as_const(vector).as_smooth() == vector);
        }

        SECTION("are_equal_") {
            // Relies on operator==, which is tested below. So just spot check.
            REQUIRE(scalar.are_equal(Smooth{}));
            REQUIRE_FALSE(vector.are_equal(matrix));
        }
    }

    SECTION("Utility methods") {
        SECTION("swap") {
            Smooth matrix_copy(matrix);
            Smooth tensor_copy(tensor);

            matrix.swap(tensor);
            REQUIRE(matrix == tensor_copy);
            REQUIRE(tensor == matrix_copy);
        }

        SECTION("operator==") {
            // Same shapes
            REQUIRE(scalar == Smooth{});
            REQUIRE(vector == Smooth{1});
            REQUIRE(matrix == Smooth{2, 3});    // Different ctor than matrix
            REQUIRE(tensor == Smooth{3, 4, 5}); // Different ctor than tensor

            // Different ranks
            REQUIRE_FALSE(scalar == vector);
            REQUIRE_FALSE(scalar == matrix);
            REQUIRE_FALSE(scalar == tensor);
            REQUIRE_FALSE(matrix == vector); // Checks low rank on rhs
            REQUIRE_FALSE(tensor == vector); // Checks low rank on rhs
            REQUIRE_FALSE(matrix == tensor);

            // Different extents (not possible for scalar)
            REQUIRE_FALSE(vector == Smooth{2});       // Completely different
            REQUIRE_FALSE(matrix == Smooth{3, 2});    // is permutation
            REQUIRE_FALSE(tensor == Smooth{6, 4, 5}); // 1st mode is different
            REQUIRE_FALSE(tensor == Smooth{3, 6, 5}); // 2nd mode is different
            REQUIRE_FALSE(tensor ==
                          Smooth{3, 4, 6}); // only last mode different
        }

        SECTION("operator!=") {
            // Implemented by negating operator==, so just spot check
            REQUIRE_FALSE(scalar != Smooth{});
            REQUIRE(scalar != vector);
        }
    }
}
