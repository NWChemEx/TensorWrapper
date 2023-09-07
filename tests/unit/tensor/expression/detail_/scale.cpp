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
 * - tensor_ ultimately call Buffer::scale, which is already known to work.
 *   Hence we only need to check that the labels and the tensors correctly get
 *   mapped to that call. The easiest way to test this is to evaluate the
 *   operation with different tensors and label combinations and ensure we get
 *   the correct answer.
 */

TEST_CASE("Scale<field::Scalar>") {
    using field_type  = field::Scalar;
    using tensor_type = TensorWrapper<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    double b{2.0};

    auto ab = a("i,j") * b;
    auto ba = b * a("i,j");

    SECTION("labels_") {
        REQUIRE(ab.labels("i,j") == "i,j");
        REQUIRE(ba.labels("i,j") == "i,j");
    }

    SECTION("tensor_") {
        SECTION("c = a * b") {
            // C starts empty so we know the buffers get mapped correctly
            tensor_type c, corr{{2.0, 4.0}, {6.0, 8.0}};
            c = ab.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
            c = ba.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("ct = a * b") {
            // Checks that c's labels get mapped to either c or a
            tensor_type c, corr{{2.0, 6.0}, {4.0, 8.0}};
            c = ab.tensor("j,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
            c = ba.tensor("j,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
    }
}
