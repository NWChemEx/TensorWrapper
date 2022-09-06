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
 * - tensor_ ultimately call Buffer::add, which is already known to work.
 *   Hence we only need to check that the labels and the tensors correctly get
 *   mapped to that call. The easiest way to test this is to evaluate the
 *   operation with different tensors and label combinations and ensure we get
 *   the correct answer.
 */

TEST_CASE("Add<field::Scalar>") {
    using field_type  = field::Scalar;
    using tensor_type = TensorWrapper<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    tensor_type b{{5.0, 6.0}, {7.0, 8.0}};

    auto apb  = a("i,j") + b("i,j");
    auto apbt = a("i,j") + b("j,i");

    SECTION("labels_") {
        REQUIRE(apb.labels("i,j") == "i,j");
        REQUIRE(apbt.labels("j,i") == "j,i");
    }

    SECTION("tensor_") {
        SECTION("c = a + b") {
            // C starts empty, so up to commuting a and b we know the buffers
            // get mapped correctly
            tensor_type c, corr{{6.0, 8.0}, {10.0, 12.0}};
            c = apb.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("c = a + bt") {
            // Checks that b's labels get mapped to either a or b
            tensor_type c, corr{{6.0, 9.0}, {9.0, 12.0}};
            c = apbt.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("c = at + b") {
            // Checks that a's labels get mapped to either a or b
            tensor_type c, corr{{6.0, 9.0}, {9.0, 12.0}};
            c = apbt.tensor("j,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("ct = a + b") {
            // Checks that c's labels get mapped to c
            tensor_type c, corr{{6.0, 10.0}, {8.0, 12.0}};
            c = apb.tensor("j,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
    }
}
