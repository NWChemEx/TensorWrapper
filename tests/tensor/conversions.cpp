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

#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("to_vector") {
    using tensor_type = tensorwrapper::tensor::ScalarTensorWrapper;
    auto tensors      = testing::get_tensors<field::Scalar>();

    SECTION("vector") {
        const auto t = tensors.at("vector");
        std::vector<double> corr{1.0, 2.0, 3.0};
        REQUIRE(to_vector(t) == corr);
    }

    SECTION("matrix") {
        const auto t = tensors.at("matrix");
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0};
        REQUIRE(to_vector(t) == corr);
    }

    SECTION("tensor") {
        const auto t = tensors.at("tensor");
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        REQUIRE(to_vector(t) == corr);
    }

    // The conversion previously had a bug that incorrectly computed the
    // flattened offset when the tensor had more than one tile. This test
    // ensures that bug doesn't come back
    SECTION("More than one tile") {
        using field_t = field::Scalar;
        using alloc_t = allocator::TiledArrayAllocator<field_t>;
        auto storage  = allocator::ta::Storage::Core;
        auto tiling   = allocator::ta::Tiling::SingleElementTile;
        auto palloc   = std::make_unique<alloc_t>(storage, tiling);
        auto t        = tensors.at("matrix");
        t.reallocate(std::move(palloc));
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0};
        REQUIRE(to_vector(t) == corr);
    }
}

TEST_CASE("Wrap std::vector") {
    using vector_il  = TA::detail::vector_il<double>;
    using double_vec = std::vector<double>;
    using ta_array   = TA::TSpArrayD;
    using twrapper   = ScalarTensorWrapper;

    auto& world = TA::get_default_world();

    vector_il v_il{1, 2, 3, 4};
    double_vec v(v_il);
    auto corr_wv = detail_::ta_to_tw(ta_array(world, v_il));

    twrapper wv = wrap_std_vector(double_vec(v_il));

    REQUIRE(wv == corr_wv);
}
