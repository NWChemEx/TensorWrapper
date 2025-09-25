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

#include "../testing/testing.hpp"
#include <tensorwrapper/shape/shape_from_labels.hpp>

using namespace tensorwrapper::shape;

TEST_CASE("shape_form_labels") {
    using shape_type = tensorwrapper::shape::Smooth;
    using label_type = tensorwrapper::shape::ShapeBase::label_type;
    shape_type s0{};
    shape_type s1{4};
    shape_type s2{5, 6};
    shape_type s3{7, 5, 4};

    SECTION("Throws if label is not found") {
        using except_t = std::runtime_error;
        label_type i("i");
        REQUIRE_THROWS_AS(shape_from_labels(i, s0("")), except_t);
        REQUIRE_THROWS_AS(shape_from_labels(i, s1("j")), except_t);
        REQUIRE_THROWS_AS(shape_from_labels(i, s1("j"), s2("k,l")), except_t);
    }

    SECTION("Scalar labels") {
        label_type empty("");
        REQUIRE(shape_from_labels(empty, s0("")) == s0);
        REQUIRE(shape_from_labels(empty, s1("i")) == s0);
        REQUIRE(shape_from_labels(empty, s1("i"), s2("j,k")) == s0);
        REQUIRE(shape_from_labels(empty, s3("i,j,k")) == s0);
    }

    SECTION("Vector labels") {
        label_type i("i"), j("j"), k("k");
        REQUIRE(shape_from_labels(i, s1("i")) == s1);
        REQUIRE(shape_from_labels(j, s2("i,j")) == shape_type({6}));
        REQUIRE(shape_from_labels(k, s2("i,j"), s3("j,k,l")) ==
                shape_type({5}));
    }

    SECTION("Matrix labels") {
        label_type ij("i,j"), jk("j,k"), ik("i,k");
        REQUIRE(shape_from_labels(ij, s2("i,j")) == s2);
        REQUIRE(shape_from_labels(jk, s3("i,j,k")) == shape_type({5, 4}));
        REQUIRE(shape_from_labels(ik, s2("i,j"), s3("j,k,l")) ==
                shape_type({5, 5}));
    }

    SECTION("Tensor labels") {
        label_type ijk("i,j,k"), ijl("i,j,l");
        REQUIRE(shape_from_labels(ijk, s3("i,j,k")) == s3);
        REQUIRE(shape_from_labels(ijl, s2("i,j"), s3("j,k,l")) ==
                shape_type({5, 6, 4}));
    }
}
