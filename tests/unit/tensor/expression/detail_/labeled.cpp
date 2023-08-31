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
#include <tensorwrapper/tensor/expression/detail_/labeled.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("Labeled<field::Scalar>") {
    using field_type  = field::Scalar;
    using tensor_type = TensorWrapper<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};

    auto lv  = a("i,j");
    auto exp = lv.expression();

    SECTION("labels_") {
        REQUIRE(exp.labels("i,j") == "i,j");
        REQUIRE(exp.labels("j,i") == "i,j");
    }

    SECTION("tensor_") {
        auto c = exp.tensor("i,j", a.shape(), a.allocator());
        REQUIRE(c == a);

        tensor_type corr{{1.0, 3.0}, {2.0, 4.0}};
        auto ct = exp.tensor("j,i", corr.shape(), corr.allocator());
        REQUIRE(ct == corr);
    }
}
