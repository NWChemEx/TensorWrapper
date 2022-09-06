/*
 * Copyright 2022 NWChemEx-Project
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

#include "../../test_tensor.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

using namespace tensorwrapper::tensor;

/* Testing Strategy
 *
 * - For classes derived from NNary we need to test that labels_ and tensor_ are
 *   implemented correctly (ctor, clone_, and are_equal_ are tested in
 *   nnary.cpp)
 * - tensor_ ultimately call Buffer::times, which is already known to
 *   work. Hence we only need to check that the labels and the tensors correctly
 *   get mapped to that call. The easiest way to test this is to evaluate the
 *   operation with different tensors and label combinations and ensure we get
 *   the correct answer.
 */

TEST_CASE("Times<field::Scalar>") {
    using field_type  = field::Scalar;
    using tensor_type = TensorWrapper<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    tensor_type b{{5.0, 6.0}, {7.0, 8.0}};

    auto axb  = a("i,j") * b("j,k");
    auto axbt = a("i,j") * b("k,j");

    SECTION("labels_") {
        REQUIRE(axb.labels("i,k") == "i,k");
        REQUIRE(axbt.labels("k,i") == "k,i");
    }

    SECTION("tensor_") {
        SECTION("c = a * b") {
            // C starts empty, so we know all the buffers get mapped correctly
            tensor_type c, corr{{19, 22}, {43, 50}};
            c = axb.tensor("i,k", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("c = a * bt") {
            // Checks that b's labels get mapped to b
            tensor_type c, corr{{17, 23}, {39, 53}};
            c = axbt.tensor("i,k", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("ct = a * b") {
            // Checks that a's labels get mapped to a
            tensor_type c, corr{{19, 43}, {22, 50}};
            c = axb.tensor("k,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("will use einsum") {
            tensor_type c, corr{{{5, 6}, {14, 16}}, {{15, 18}, {28, 32}}};
            c = axb.tensor("i,j,k", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
    }
}
