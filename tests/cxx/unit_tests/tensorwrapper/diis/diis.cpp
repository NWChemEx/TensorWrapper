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
#include <tensorwrapper/diis/diis.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>

using diis_type     = tensorwrapper::diis::DIIS;
using tensor_type   = tensorwrapper::Tensor;
using elements_type = std::vector<double>;

template<typename FloatType>
tensor_type make_tensor(elements_type elems) {
    auto pbuffer = tensorwrapper::testing::eigen_matrix<FloatType>(2, 2);
    pbuffer->set_elem({0, 0}, elems[0]);
    pbuffer->set_elem({0, 1}, elems[1]);
    pbuffer->set_elem({1, 0}, elems[2]);
    pbuffer->set_elem({1, 1}, elems[3]);
    auto pshape = pbuffer->layout().shape().clone();
    return tensor_type(std::move(pshape), std::move(pbuffer));
}

TEMPLATE_LIST_TEST_CASE("DIIS", "",
                        tensorwrapper::types::floating_point_types) {
    // Inputs
    tensor_type i1 = make_tensor<TestType>({1.0, 2.0, 3.0, 4.0});
    tensor_type i2 = make_tensor<TestType>({6.0, 5.0, 8.0, 7.0});
    tensor_type i3 = make_tensor<TestType>({12.0, 11.0, 10.0, 9.0});

    SECTION("Typedefs") {
        SECTION("size_type") {
            using corr_t = std::size_t;
            using the_t  = diis_type::size_type;
            STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
        }
        SECTION("tensor_type") {
            using corr_t = tensor_type;
            using the_t  = diis_type::tensor_type;
            STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
        }
    }

    SECTION("Comparisons") {
        auto defaulted        = diis_type();
        auto two_samples_max  = diis_type(2);
        auto extrapolate_used = diis_type();
        auto temp             = extrapolate_used.extrapolate(i1, i3);
        SECTION("Equals") {
            REQUIRE(defaulted == diis_type());
            REQUIRE(two_samples_max == diis_type(2));
        }
        SECTION("Max samples not equal") {
            REQUIRE(two_samples_max != defaulted);
        }
        SECTION("Recorded values different") {
            REQUIRE(defaulted != extrapolate_used);
        }
    }

    SECTION("extrapolate") {
        // Outputs
        tensor_type corr1 = make_tensor<TestType>({1.0, 2.0, 3.0, 4.0});
        tensor_type corr2 = make_tensor<TestType>({12.0, 8.6, 14.0, 10.6});
        tensor_type corr3 = make_tensor<TestType>(
          {15.35294118, 14.35294118, 11.11764706, 10.11764706});

        // Call extrapolate enough to require removing an old value
        auto diis    = diis_type(2);
        auto output1 = diis.extrapolate(i1, i3);
        auto output2 = diis.extrapolate(i2, i2);
        auto output3 = diis.extrapolate(i3, i1);

        using tensorwrapper::operations::approximately_equal;
        REQUIRE(approximately_equal(output1, corr1, 1E-6));
        REQUIRE(approximately_equal(output2, corr2, 1E-6));
        REQUIRE(approximately_equal(output3, corr3, 1E-6));
    }
}
