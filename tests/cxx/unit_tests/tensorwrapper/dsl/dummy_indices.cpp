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
#include <tensorwrapper/dsl/dummy_indices.hpp>

using namespace tensorwrapper;

TEST_CASE("DummyIndices<std::string>") {
    using dummy_indices_type = dsl::DummyIndices<std::string>;

    dummy_indices_type defaulted;
    dummy_indices_type scalar("");
    dummy_indices_type vector("i");
    dummy_indices_type matrix("i, j");
    dummy_indices_type tensor("i, jk, l");

    SECTION("CTors") {
        SECTION("defaulted") { REQUIRE(defaulted.size() == 0); }

        SECTION("string value") {
            REQUIRE(scalar.size() == 0);

            REQUIRE(vector.size() == 1);
            REQUIRE(vector[0] == "i");

            REQUIRE(matrix.size() == 2);
            REQUIRE(matrix[0] == "i");
            REQUIRE(matrix[1] == "j");

            REQUIRE(tensor.size() == 3);
            REQUIRE(tensor[0] == "i");
            REQUIRE(tensor[1] == "jk");
            REQUIRE(tensor[2] == "l");

            // Dummy indices can't be empty
            REQUIRE_THROWS_AS(dummy_indices_type("i, "), std::runtime_error);
        }

        testing::test_copy_move_ctor_and_assignment(defaulted, scalar, vector,
                                                    matrix, tensor);
    }

    SECTION("unique_index_size") {
        REQUIRE(defaulted.unique_index_size() == 0);
        REQUIRE(scalar.unique_index_size() == 0);
        REQUIRE(vector.unique_index_size() == 1);
        REQUIRE(matrix.unique_index_size() == 2);
        REQUIRE(tensor.unique_index_size() == 3);
        REQUIRE(dummy_indices_type("i,i").unique_index_size() == 1);
    }

    SECTION("has_repeated_indices") {
        REQUIRE_FALSE(defaulted.has_repeated_indices());
        REQUIRE_FALSE(scalar.has_repeated_indices());
        REQUIRE_FALSE(vector.has_repeated_indices());
        REQUIRE_FALSE(matrix.has_repeated_indices());
        REQUIRE_FALSE(tensor.has_repeated_indices());
        REQUIRE(dummy_indices_type("i,i").has_repeated_indices());
    }

    SECTION("is_permutation") {
        REQUIRE(scalar.is_permutation(scalar));
        REQUIRE_FALSE(scalar.is_permutation(vector));

        REQUIRE(vector.is_permutation(vector));
        REQUIRE_FALSE(vector.is_permutation(scalar));
        REQUIRE_FALSE(vector.is_permutation(dummy_indices_type("j")));

        REQUIRE(matrix.is_permutation(matrix));
        REQUIRE(matrix.is_permutation(dummy_indices_type("j,i")));
        REQUIRE_FALSE(matrix.is_permutation(scalar));
        REQUIRE_FALSE(matrix.is_permutation(dummy_indices_type("i,k")));
    }

    SECTION("permutation") {
        using offset_vector = typename dummy_indices_type::offset_vector;

        REQUIRE(scalar.permutation(scalar) == offset_vector{});

        REQUIRE(vector.permutation(vector) == offset_vector{0});

        dummy_indices_type matrix2("j,i");
        REQUIRE(matrix.permutation(matrix) == offset_vector{0, 1});
        REQUIRE(matrix.permutation(matrix2) == offset_vector{1, 0});
        REQUIRE(matrix2.permutation(matrix) == offset_vector{1, 0});

        dummy_indices_type tensor2("jk, i, l");
        dummy_indices_type tensor3("l, jk, i");
        dummy_indices_type tensor4("i,l,jk");
        dummy_indices_type tensor5("l,i,jk");
        dummy_indices_type tensor6("jk, l, i");
        REQUIRE(tensor.permutation(tensor) == offset_vector{0, 1, 2});
        REQUIRE(tensor.permutation(tensor2) == offset_vector{1, 0, 2});
        REQUIRE(tensor.permutation(tensor3) == offset_vector{2, 1, 0});
        REQUIRE(tensor.permutation(tensor4) == offset_vector{0, 2, 1});
        REQUIRE(tensor.permutation(tensor5) == offset_vector{1, 2, 0});
        REQUIRE(tensor.permutation(tensor6) == offset_vector{2, 0, 1});

        dummy_indices_type repeated("i,i");

        // Must have same number of indices
        REQUIRE_THROWS_AS(scalar.permutation(vector), std::runtime_error);

        // *this can't have repeated indics
        REQUIRE_THROWS_AS(repeated.permutation(matrix), std::runtime_error);

        // other can't have repeated indices
        REQUIRE_THROWS_AS(matrix.permutation(repeated), std::runtime_error);

        // error if index isn't in both
        dummy_indices_type other("j");

        REQUIRE_THROWS_AS(vector.permutation(other), std::runtime_error);
    }

    SECTION("find(const_reference)") {
        using offset_vector = typename dummy_indices_type::offset_vector;
        REQUIRE(defaulted.find("") == offset_vector{});

        REQUIRE(scalar.find("") == offset_vector{});

        REQUIRE(vector.find("i") == offset_vector{0});
        REQUIRE(vector.find("j") == offset_vector{});

        REQUIRE(matrix.find("i") == offset_vector{0});
        REQUIRE(matrix.find("j") == offset_vector{1});

        REQUIRE(tensor.find("i") == offset_vector{0});
        REQUIRE(tensor.find("jk") == offset_vector{1});
        REQUIRE(tensor.find("l") == offset_vector{2});

        REQUIRE(dummy_indices_type("i,i").find("i") == offset_vector{0, 1});
    }

    SECTION("count") {
        REQUIRE(defaulted.count("") == 0);

        REQUIRE(scalar.count("") == 0);

        REQUIRE(vector.count("") == 0);
        REQUIRE(vector.count("i") == 1);
        REQUIRE(vector.count("j") == 0);

        REQUIRE(matrix.count("") == 0);
        REQUIRE(matrix.count("i") == 1);
        REQUIRE(matrix.count("j") == 1);
        REQUIRE(dummy_indices_type("i,i").count("i") == 2);
    }

    SECTION("operator==") {
        // Default construction is indistinguishable from scalar indices
        REQUIRE(defaulted == scalar);
        REQUIRE(defaulted == "");

        // Different ranks are different
        REQUIRE_FALSE(defaulted == vector);
        REQUIRE_FALSE(defaulted == "i");

        // Same vector indices
        REQUIRE(vector == dummy_indices_type("i"));
        REQUIRE(vector == "i");

        // Different vector indices
        REQUIRE_FALSE(vector == dummy_indices_type("j"));
        REQUIRE_FALSE(vector == "j");

        // Same matrix indices
        REQUIRE(matrix == dummy_indices_type("i,j"));
        REQUIRE(matrix == "i,j");

        // Spaces aren't significant
        REQUIRE(matrix == dummy_indices_type("i, j"));
        REQUIRE(matrix == "i, j");
        REQUIRE(matrix == dummy_indices_type(" i , j "));
        REQUIRE(matrix == " i , j ");

        // Are case sensitive
        REQUIRE_FALSE(matrix == dummy_indices_type("I,j"));
        REQUIRE_FALSE(matrix == "I,j");

        // Permutations are different
        REQUIRE_FALSE(matrix == dummy_indices_type("j,i"));
        REQUIRE_FALSE(matrix == "j,i");
    }

    SECTION("operator!=") {
        // Just negates operator== so spot checking is fine
        REQUIRE_FALSE(vector != dummy_indices_type("i"));
        REQUIRE_FALSE(vector != "i");
        REQUIRE(vector != dummy_indices_type("j"));
        REQUIRE(vector != "j");
    }

    SECTION("concatenation") {
        dummy_indices_type matrix2("k,l");
        REQUIRE(scalar.concatenation(scalar) == dummy_indices_type(""));
        REQUIRE(scalar.concatenation(vector) == dummy_indices_type("i"));
        REQUIRE(scalar.concatenation(matrix) == dummy_indices_type("i,j"));
        REQUIRE(scalar.concatenation(matrix2) == dummy_indices_type("k,l"));

        REQUIRE(vector.concatenation(scalar) == dummy_indices_type("i"));
        REQUIRE(vector.concatenation(vector) == dummy_indices_type("i,i"));
        REQUIRE(vector.concatenation(matrix) == dummy_indices_type("i,i,j"));
        REQUIRE(vector.concatenation(matrix2) == dummy_indices_type("i,k,l"));

        REQUIRE(matrix.concatenation(scalar) == dummy_indices_type("i,j"));
        REQUIRE(matrix.concatenation(vector) == dummy_indices_type("i,j,i"));
        REQUIRE(matrix.concatenation(matrix) == dummy_indices_type("i,j,i,j"));
        REQUIRE(matrix.concatenation(matrix2) == dummy_indices_type("i,j,k,l"));
    }

    SECTION("intersection") {
        dummy_indices_type matrix2("k,l");
        REQUIRE(scalar.intersection(scalar) == dummy_indices_type(""));
        REQUIRE(scalar.intersection(vector) == dummy_indices_type(""));
        REQUIRE(scalar.intersection(matrix) == dummy_indices_type(""));
        REQUIRE(scalar.intersection(matrix2) == dummy_indices_type(""));

        REQUIRE(vector.intersection(scalar) == dummy_indices_type(""));
        REQUIRE(vector.intersection(vector) == dummy_indices_type("i"));
        REQUIRE(vector.intersection(matrix) == dummy_indices_type("i"));
        REQUIRE(vector.intersection(matrix2) == dummy_indices_type(""));

        REQUIRE(matrix.intersection(scalar) == dummy_indices_type(""));
        REQUIRE(matrix.intersection(vector) == dummy_indices_type("i"));
        REQUIRE(matrix.intersection(matrix) == dummy_indices_type("i,j"));
        REQUIRE(matrix.intersection(matrix2) == dummy_indices_type(""));
    }
}
