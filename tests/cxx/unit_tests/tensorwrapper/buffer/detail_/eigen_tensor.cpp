/*
 * Copyright 2025 NWChemEx-Project
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

#include "../../testing/testing.hpp"
#include <iomanip>
#include <tensorwrapper/buffer/detail_/eigen_tensor.hpp>

using namespace tensorwrapper;
using namespace testing;

using buffer::detail_::hash_utilities::hash_input;

template<typename FloatType, unsigned int Rank>
using pimpl_type = buffer::detail_::EigenTensor<FloatType, Rank>;
using shape_type = shape::Smooth;

// Should be the same regardless of template parameters
using label_type = typename pimpl_type<double, 0>::label_type;
using hash_type  = typename pimpl_type<double, 0>::hash_type;

TEMPLATE_LIST_TEST_CASE("EigenTensor", "", types::floating_point_types) {
    pimpl_type<TestType, 0> scalar(shape_type{});
    scalar.get_elem({}) = 1.0;

    pimpl_type<TestType, 1> vector(shape_type{2});
    vector.get_elem({0}) = 1.0;
    vector.get_elem({1}) = 2.0;

    pimpl_type<TestType, 2> matrix(shape_type{2, 2});
    matrix.get_elem({0, 0}) = 1.0;
    matrix.get_elem({0, 1}) = 2.0;
    matrix.get_elem({1, 0}) = 3.0;
    matrix.get_elem({1, 1}) = 4.0;

    pimpl_type<TestType, 3> tensor(shape_type{2, 2, 2});
    tensor.get_elem({0, 0, 0}) = 1.0;
    tensor.get_elem({0, 0, 1}) = 2.0;
    tensor.get_elem({0, 1, 0}) = 3.0;
    tensor.get_elem({0, 1, 1}) = 4.0;
    tensor.get_elem({1, 0, 0}) = 5.0;
    tensor.get_elem({1, 0, 1}) = 6.0;
    tensor.get_elem({1, 1, 0}) = 7.0;
    tensor.get_elem({1, 1, 1}) = 8.0;

    // -------------------------------------------------------------------------
    // -- Public methods
    // -------------------------------------------------------------------------

    SECTION("operator==") {
        SECTION("Same State") {
            pimpl_type<TestType, 0> scalar2(scalar);
            REQUIRE(scalar2 == scalar);
        }

        SECTION("Different Value") {
            pimpl_type<TestType, 0> scalar2(scalar);
            scalar2.get_elem({}) = 42.0;
            REQUIRE_FALSE(scalar2 == scalar);
            // Ensure hash is recalculated after change
            *(scalar2.data()) = 1.0;
            REQUIRE(scalar2 == scalar);
        }

        SECTION("Different Extents") {
            pimpl_type<TestType, 1> vector2(shape_type{1});
            vector.get_elem({0}) = 1.0;
            REQUIRE_FALSE(vector2 == vector);
        }

        if constexpr(types::is_uncertain_v<TestType>) {
            SECTION("Check Error Sources Match") {
                pimpl_type<TestType, 0> uscalar(shape_type{});
                uscalar.get_elem({}) = TestType(1.0, 0.0);
                pimpl_type<TestType, 0> uscalar2(uscalar);
                REQUIRE(uscalar2 == uscalar);
            }
        }
    }

    SECTION("get_hash") {
        SECTION("scalar") {
            hash_type scalar_hash = scalar.get_hash();

            hash_type corr{std::as_const(scalar).rank()};
            hash_input(corr, std::as_const(scalar).get_elem({}));
            REQUIRE(scalar_hash == corr);
        }
        SECTION("vector") {
            hash_type vector_hash = vector.get_hash();

            using buffer::detail_::hash_utilities::hash_input;
            hash_type corr{std::as_const(vector).rank()};
            hash_input(corr, std::as_const(vector).extent(0));
            hash_input(corr, std::as_const(vector).get_elem({0}));
            hash_input(corr, std::as_const(vector).get_elem({1}));
            REQUIRE(vector_hash == corr);
        }
    }

    // -------------------------------------------------------------------------
    // -- Protected methods
    // -------------------------------------------------------------------------

    SECTION("clone_") {
        REQUIRE(scalar.clone()->are_equal(scalar));
        REQUIRE(vector.clone()->are_equal(vector));
        REQUIRE(matrix.clone()->are_equal(matrix));
        REQUIRE(tensor.clone()->are_equal(tensor));
    }

    SECTION("rank_") {
        REQUIRE(scalar.rank() == 0);
        REQUIRE(vector.rank() == 1);
        REQUIRE(matrix.rank() == 2);
        REQUIRE(tensor.rank() == 3);

        pimpl_type<TestType, 6> defaulted;
        REQUIRE(defaulted.rank() == 6);
    }

    SECTION("extent_") {
        REQUIRE(vector.extent(0) == 2);

        REQUIRE(matrix.extent(0) == 2);
        REQUIRE(matrix.extent(1) == 2);

        REQUIRE(tensor.extent(0) == 2);
        REQUIRE(tensor.extent(1) == 2);
        REQUIRE(tensor.extent(2) == 2);
    }

    SECTION("data_()") {
        REQUIRE(*scalar.data() == TestType{1.0});

        REQUIRE(*vector.data() == TestType{1.0});
        REQUIRE(*(vector.data() + 1) == TestType{2.0});

        REQUIRE(*matrix.data() == TestType{1.0});
        REQUIRE(*(matrix.data() + 1) == TestType{2.0});
        REQUIRE(*(matrix.data() + 2) == TestType{3.0});
        REQUIRE(*(matrix.data() + 3) == TestType{4.0});

        REQUIRE(*tensor.data() == TestType{1.0});
        REQUIRE(*(tensor.data() + 1) == TestType{2.0});
        REQUIRE(*(tensor.data() + 2) == TestType{3.0});
        REQUIRE(*(tensor.data() + 3) == TestType{4.0});
        REQUIRE(*(tensor.data() + 4) == TestType{5.0});
        REQUIRE(*(tensor.data() + 5) == TestType{6.0});
        REQUIRE(*(tensor.data() + 6) == TestType{7.0});
        REQUIRE(*(tensor.data() + 7) == TestType{8.0});
    }

    SECTION("data_() const") {
        REQUIRE(*std::as_const(scalar).data() == TestType{1.0});

        REQUIRE(*std::as_const(vector).data() == TestType{1.0});
        REQUIRE(*(std::as_const(vector).data() + 1) == TestType{2.0});

        REQUIRE(*std::as_const(matrix).data() == TestType{1.0});
        REQUIRE(*(std::as_const(matrix).data() + 1) == TestType{2.0});
        REQUIRE(*(std::as_const(matrix).data() + 2) == TestType{3.0});
        REQUIRE(*(std::as_const(matrix).data() + 3) == TestType{4.0});

        REQUIRE(*std::as_const(tensor).data() == TestType{1.0});
        REQUIRE(*(std::as_const(tensor).data() + 1) == TestType{2.0});
        REQUIRE(*(std::as_const(tensor).data() + 2) == TestType{3.0});
        REQUIRE(*(std::as_const(tensor).data() + 3) == TestType{4.0});
        REQUIRE(*(std::as_const(tensor).data() + 4) == TestType{5.0});
        REQUIRE(*(std::as_const(tensor).data() + 5) == TestType{6.0});
        REQUIRE(*(std::as_const(tensor).data() + 6) == TestType{7.0});
        REQUIRE(*(std::as_const(tensor).data() + 7) == TestType{8.0});
    }

    SECTION("get_elem_ ()") {
        REQUIRE(scalar.get_elem({}) == TestType{1.0});

        REQUIRE(vector.get_elem({0}) == TestType{1.0});
        REQUIRE(vector.get_elem({1}) == TestType{2.0});

        REQUIRE(matrix.get_elem({0, 0}) == TestType{1.0});
        REQUIRE(matrix.get_elem({0, 1}) == TestType{2.0});
        REQUIRE(matrix.get_elem({1, 0}) == TestType{3.0});
        REQUIRE(matrix.get_elem({1, 1}) == TestType{4.0});

        REQUIRE(tensor.get_elem({0, 0, 0}) == TestType{1.0});
        REQUIRE(tensor.get_elem({0, 0, 1}) == TestType{2.0});
        REQUIRE(tensor.get_elem({0, 1, 0}) == TestType{3.0});
        REQUIRE(tensor.get_elem({0, 1, 1}) == TestType{4.0});
        REQUIRE(tensor.get_elem({1, 0, 0}) == TestType{5.0});
        REQUIRE(tensor.get_elem({1, 0, 1}) == TestType{6.0});
        REQUIRE(tensor.get_elem({1, 1, 0}) == TestType{7.0});
        REQUIRE(tensor.get_elem({1, 1, 1}) == TestType{8.0});
    }

    SECTION("get_elem_() const") {
        REQUIRE(std::as_const(scalar).get_elem({}) == TestType{1.0});

        REQUIRE(std::as_const(vector).get_elem({0}) == TestType{1.0});
        REQUIRE(std::as_const(vector).get_elem({1}) == TestType{2.0});

        REQUIRE(std::as_const(matrix).get_elem({0, 0}) == TestType{1.0});
        REQUIRE(std::as_const(matrix).get_elem({0, 1}) == TestType{2.0});
        REQUIRE(std::as_const(matrix).get_elem({1, 0}) == TestType{3.0});
        REQUIRE(std::as_const(matrix).get_elem({1, 1}) == TestType{4.0});

        REQUIRE(std::as_const(tensor).get_elem({0, 0, 0}) == TestType{1.0});
        REQUIRE(std::as_const(tensor).get_elem({0, 0, 1}) == TestType{2.0});
        REQUIRE(std::as_const(tensor).get_elem({0, 1, 0}) == TestType{3.0});
        REQUIRE(std::as_const(tensor).get_elem({0, 1, 1}) == TestType{4.0});
        REQUIRE(std::as_const(tensor).get_elem({1, 0, 0}) == TestType{5.0});
        REQUIRE(std::as_const(tensor).get_elem({1, 0, 1}) == TestType{6.0});
        REQUIRE(std::as_const(tensor).get_elem({1, 1, 0}) == TestType{7.0});
        REQUIRE(std::as_const(tensor).get_elem({1, 1, 1}) == TestType{8.0});
    }

    SECTION("are_equal_") {
        pimpl_type<TestType, 0> scalar2(scalar);
        REQUIRE(scalar2.are_equal(scalar));

        scalar2.get_elem({}) = 42.0;
        REQUIRE_FALSE(scalar2.are_equal(scalar));
    }

    SECTION("to_string_") {
        std::stringstream sone;
        sone << TestType{1.0};

        std::stringstream stwo;
        stwo << TestType{2.0};

        REQUIRE(scalar.to_string() == sone.str());
        REQUIRE(vector.to_string() == sone.str() + " " + stwo.str());
    }

    SECTION("add_to_stream_") {
        std::stringstream ss, ss_corr;
        ss << std::fixed << std::setprecision(4);
        scalar.add_to_stream(ss);
        ss_corr << std::fixed << std::setprecision(4);
        ss_corr << TestType{1.0};
        REQUIRE(ss.str() == ss_corr.str());
        REQUIRE_FALSE(ss.str() == scalar.to_string());
    }

    SECTION("addition_assignment_") {
        SECTION("scalar") {
            pimpl_type<TestType, 0> output;
            label_type s("");
            output.addition_assignment(s, s, s, scalar, scalar);

            pimpl_type<TestType, 0> corr(shape_type{});
            corr.get_elem({}) = 2.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute none") {
            pimpl_type<TestType, 3> output;
            label_type o("i,j,k");
            label_type l("i,j,k");
            label_type r("i,j,k");

            output.addition_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 2.0;
            corr.get_elem({0, 0, 1}) = 4.0;
            corr.get_elem({0, 1, 0}) = 6.0;
            corr.get_elem({0, 1, 1}) = 8.0;
            corr.get_elem({1, 0, 0}) = 10.0;
            corr.get_elem({1, 0, 1}) = 12.0;
            corr.get_elem({1, 1, 0}) = 14.0;
            corr.get_elem({1, 1, 1}) = 16.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute LHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("k,j,i");

            output.addition_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 2.0;
            corr.get_elem({0, 0, 1}) = 7.0;
            corr.get_elem({0, 1, 0}) = 6.0;
            corr.get_elem({0, 1, 1}) = 11.0;
            corr.get_elem({1, 0, 0}) = 7.0;
            corr.get_elem({1, 0, 1}) = 12.0;
            corr.get_elem({1, 1, 0}) = 11.0;
            corr.get_elem({1, 1, 1}) = 16.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute RHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("k,j,i");
            label_type r("i,j,k");

            output.addition_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 2.0;
            corr.get_elem({0, 0, 1}) = 7.0;
            corr.get_elem({0, 1, 0}) = 6.0;
            corr.get_elem({0, 1, 1}) = 11.0;
            corr.get_elem({1, 0, 0}) = 7.0;
            corr.get_elem({1, 0, 1}) = 12.0;
            corr.get_elem({1, 1, 0}) = 11.0;
            corr.get_elem({1, 1, 1}) = 16.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute all") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("j,i,k");

            output.addition_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 2.0;
            corr.get_elem({0, 0, 1}) = 8.0;
            corr.get_elem({0, 1, 0}) = 8.0;
            corr.get_elem({0, 1, 1}) = 14.0;
            corr.get_elem({1, 0, 0}) = 4.0;
            corr.get_elem({1, 0, 1}) = 10.0;
            corr.get_elem({1, 1, 0}) = 10.0;
            corr.get_elem({1, 1, 1}) = 16.0;
            REQUIRE(output == corr);
        }
    }

    SECTION("subtraction_assignment_") {
        SECTION("scalar") {
            pimpl_type<TestType, 0> output;
            label_type s("");
            output.subtraction_assignment(s, s, s, scalar, scalar);

            pimpl_type<TestType, 0> corr(shape_type{});
            corr.get_elem({}) = 0.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute none") {
            pimpl_type<TestType, 3> output;
            label_type o("i,j,k");
            label_type l("i,j,k");
            label_type r("i,j,k");

            output.subtraction_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 0.0;
            corr.get_elem({0, 0, 1}) = 0.0;
            corr.get_elem({0, 1, 0}) = 0.0;
            corr.get_elem({0, 1, 1}) = 0.0;
            corr.get_elem({1, 0, 0}) = 0.0;
            corr.get_elem({1, 0, 1}) = 0.0;
            corr.get_elem({1, 1, 0}) = 0.0;
            corr.get_elem({1, 1, 1}) = 0.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute LHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("k,j,i");

            output.subtraction_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 0.0;
            corr.get_elem({0, 0, 1}) = 3.0;
            corr.get_elem({0, 1, 0}) = 0.0;
            corr.get_elem({0, 1, 1}) = 3.0;
            corr.get_elem({1, 0, 0}) = -3.0;
            corr.get_elem({1, 0, 1}) = 0.0;
            corr.get_elem({1, 1, 0}) = -3.0;
            corr.get_elem({1, 1, 1}) = 0.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute RHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("k,j,i");
            label_type r("i,j,k");

            output.subtraction_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 0.0;
            corr.get_elem({0, 0, 1}) = -3.0;
            corr.get_elem({0, 1, 0}) = 0.0;
            corr.get_elem({0, 1, 1}) = -3.0;
            corr.get_elem({1, 0, 0}) = 3.0;
            corr.get_elem({1, 0, 1}) = 0.0;
            corr.get_elem({1, 1, 0}) = 3.0;
            corr.get_elem({1, 1, 1}) = 0.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute all") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("j,i,k");

            output.subtraction_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 0.0;
            corr.get_elem({0, 0, 1}) = 2.0;
            corr.get_elem({0, 1, 0}) = -2.0;
            corr.get_elem({0, 1, 1}) = 0.0;
            corr.get_elem({1, 0, 0}) = 0.0;
            corr.get_elem({1, 0, 1}) = 2.0;
            corr.get_elem({1, 1, 0}) = -2.0;
            corr.get_elem({1, 1, 1}) = 0.0;
            REQUIRE(output == corr);
        }
    }

    SECTION("hadamard_assignment_") {
        SECTION("scalar") {
            pimpl_type<TestType, 0> output;
            label_type s("");
            output.hadamard_assignment(s, s, s, scalar, scalar);

            pimpl_type<TestType, 0> corr(shape_type{});
            corr.get_elem({}) = 1.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute none") {
            pimpl_type<TestType, 3> output;
            label_type o("i,j,k");
            label_type l("i,j,k");
            label_type r("i,j,k");

            output.hadamard_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 1.0;
            corr.get_elem({0, 0, 1}) = 4.0;
            corr.get_elem({0, 1, 0}) = 9.0;
            corr.get_elem({0, 1, 1}) = 16.0;
            corr.get_elem({1, 0, 0}) = 25.0;
            corr.get_elem({1, 0, 1}) = 36.0;
            corr.get_elem({1, 1, 0}) = 49.0;
            corr.get_elem({1, 1, 1}) = 64.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute LHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("k,j,i");

            output.hadamard_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 1.0;
            corr.get_elem({0, 0, 1}) = 10.0;
            corr.get_elem({0, 1, 0}) = 9.0;
            corr.get_elem({0, 1, 1}) = 28.0;
            corr.get_elem({1, 0, 0}) = 10.0;
            corr.get_elem({1, 0, 1}) = 36.0;
            corr.get_elem({1, 1, 0}) = 28.0;
            corr.get_elem({1, 1, 1}) = 64.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute RHS") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("k,j,i");
            label_type r("i,j,k");

            output.hadamard_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 1.0;
            corr.get_elem({0, 0, 1}) = 10.0;
            corr.get_elem({0, 1, 0}) = 9.0;
            corr.get_elem({0, 1, 1}) = 28.0;
            corr.get_elem({1, 0, 0}) = 10.0;
            corr.get_elem({1, 0, 1}) = 36.0;
            corr.get_elem({1, 1, 0}) = 28.0;
            corr.get_elem({1, 1, 1}) = 64.0;
            REQUIRE(output == corr);
        }

        SECTION("tensor : permute all") {
            pimpl_type<TestType, 3> output;
            label_type o("k,j,i");
            label_type l("i,j,k");
            label_type r("j,i,k");

            output.hadamard_assignment(o, l, r, tensor, tensor);

            pimpl_type<TestType, 3> corr(shape_type{2, 2, 2});
            corr.get_elem({0, 0, 0}) = 1.0;
            corr.get_elem({0, 0, 1}) = 15.0;
            corr.get_elem({0, 1, 0}) = 15.0;
            corr.get_elem({0, 1, 1}) = 49.0;
            corr.get_elem({1, 0, 0}) = 4.0;
            corr.get_elem({1, 0, 1}) = 24.0;
            corr.get_elem({1, 1, 0}) = 24.0;
            corr.get_elem({1, 1, 1}) = 64.0;
            REQUIRE(output == corr);
        }
    }

    SECTION("contraction_assignment") {
        SECTION("ijk,ijk->") {
            pimpl_type<TestType, 0> output;

            label_type o("");
            label_type l("i,j,k");
            label_type r("i,j,k");
            shape_type oshape{};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 0> corr(oshape);
            corr.get_elem({}) = 204.0;
            REQUIRE(output == corr);
        }

        SECTION("ijk,jik->") {
            pimpl_type<TestType, 0> output;

            label_type o("");
            label_type l("i,j,k");
            label_type r("j,i,k");
            shape_type oshape{};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 0> corr(oshape);
            corr.get_elem({}) = 196.0;
            REQUIRE(output == corr);
        }

        SECTION("ijk,jkl->il") {
            pimpl_type<TestType, 2> output;

            label_type o("i,l");
            label_type l("i,j,k");
            label_type r("j,k,l");
            shape_type oshape{2, 2};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 2> corr(oshape);
            corr.get_elem({0, 0}) = 50.0;
            corr.get_elem({0, 1}) = 60.0;
            corr.get_elem({1, 0}) = 114.0;
            corr.get_elem({1, 1}) = 140.0;
            REQUIRE(output == corr);
        }

        SECTION("ijk,jlk->il") {
            pimpl_type<TestType, 2> output;

            label_type o("i,l");
            label_type l("i,j,k");
            label_type r("j,l,k");
            shape_type oshape{2, 2};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 2> corr(oshape);
            corr.get_elem({0, 0}) = 44.0;
            corr.get_elem({0, 1}) = 64.0;
            corr.get_elem({1, 0}) = 100.0;
            corr.get_elem({1, 1}) = 152.0;
            REQUIRE(output == corr);
        }

        SECTION("ijk,jlk->li") {
            pimpl_type<TestType, 2> output;

            label_type o("l,i");
            label_type l("i,j,k");
            label_type r("j,l,k");
            shape_type oshape{2, 2};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 2> corr(oshape);
            corr.get_elem({0, 0}) = 44.0;
            corr.get_elem({0, 1}) = 100.0;
            corr.get_elem({1, 0}) = 64.0;
            corr.get_elem({1, 1}) = 152.0;
            REQUIRE(output == corr);
        }

        SECTION("ijk,ljm->iklm") {
            pimpl_type<TestType, 4> output;

            label_type o("i,k,l,m");
            label_type l("i,j,k");
            label_type r("l,j,m");
            shape_type oshape{2, 2, 2, 2};
            output.contraction_assignment(o, l, r, oshape, tensor, tensor);

            pimpl_type<TestType, 4> corr(oshape);
            corr.get_elem({0, 0, 0, 0}) = 10.0;
            corr.get_elem({0, 0, 0, 1}) = 14.0;
            corr.get_elem({0, 0, 1, 0}) = 26.0;
            corr.get_elem({0, 0, 1, 1}) = 30.0;
            corr.get_elem({0, 1, 0, 0}) = 14.0;
            corr.get_elem({0, 1, 0, 1}) = 20.0;
            corr.get_elem({0, 1, 1, 0}) = 38.0;
            corr.get_elem({0, 1, 1, 1}) = 44.0;
            corr.get_elem({1, 0, 0, 0}) = 26.0;
            corr.get_elem({1, 0, 0, 1}) = 38.0;
            corr.get_elem({1, 0, 1, 0}) = 74.0;
            corr.get_elem({1, 0, 1, 1}) = 86.0;
            corr.get_elem({1, 1, 0, 0}) = 30.0;
            corr.get_elem({1, 1, 0, 1}) = 44.0;
            corr.get_elem({1, 1, 1, 0}) = 86.0;
            corr.get_elem({1, 1, 1, 1}) = 100.0;

            REQUIRE(output == corr);
        }

        SECTION("ij,jkl->ikl") {
            pimpl_type<TestType, 3> output;

            label_type o("i,k,l");
            label_type l("i,j");
            label_type r("j,k,l");
            shape_type oshape{2, 2, 2};
            output.contraction_assignment(o, l, r, oshape, matrix, tensor);

            pimpl_type<TestType, 3> corr(oshape);
            corr.get_elem({0, 0, 0}) = 11.0;
            corr.get_elem({0, 0, 1}) = 14.0;
            corr.get_elem({0, 1, 0}) = 17.0;
            corr.get_elem({0, 1, 1}) = 20.0;
            corr.get_elem({1, 0, 0}) = 23.0;
            corr.get_elem({1, 0, 1}) = 30.0;
            corr.get_elem({1, 1, 0}) = 37.0;
            corr.get_elem({1, 1, 1}) = 44.0;

            REQUIRE(corr == output);
        }
    }

    SECTION("permute_assignment") {
        pimpl_type<TestType, 2> output;

        SECTION("matrix : no permute") {
            label_type o("i,j");
            label_type i("i,j");
            output.permute_assignment(o, i, matrix);

            REQUIRE(output == matrix);
        }

        SECTION("matrix : permute") {
            label_type o("i,j");
            label_type i("j,i");
            output.permute_assignment(o, i, matrix);

            pimpl_type<TestType, 2> corr(shape_type{2, 2});
            corr.get_elem({0, 0}) = 1.0;
            corr.get_elem({0, 1}) = 3.0;
            corr.get_elem({1, 0}) = 2.0;
            corr.get_elem({1, 1}) = 4.0;
            REQUIRE(output == corr);
        }
    }

    SECTION("scalar_multiplication") {
        pimpl_type<TestType, 2> output;

        SECTION("matrix : no permute") {
            label_type o("i,j");
            label_type i("i,j");
            output.scalar_multiplication(o, i, 2.0, matrix);

            pimpl_type<TestType, 2> corr(shape_type{2, 2});
            corr.get_elem({0, 0}) = 2.0;
            corr.get_elem({0, 1}) = 4.0;
            corr.get_elem({1, 0}) = 6.0;
            corr.get_elem({1, 1}) = 8.0;

            REQUIRE(output == corr);
        }

        SECTION("matrix : permute") {
            label_type o("i,j");
            label_type i("j,i");
            output.scalar_multiplication(o, i, 2.0, matrix);

            pimpl_type<TestType, 2> corr(shape_type{2, 2});
            corr.get_elem({0, 0}) = 2.0;
            corr.get_elem({0, 1}) = 6.0;
            corr.get_elem({1, 0}) = 4.0;
            corr.get_elem({1, 1}) = 8.0;
            REQUIRE(output == corr);
        }
    }
}
