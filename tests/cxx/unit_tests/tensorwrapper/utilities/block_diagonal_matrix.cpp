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
#include <tensorwrapper/utilities/block_diagonal_matrix.hpp>

using namespace tensorwrapper;
using namespace testing;

using tensorwrapper::utilities::block_diagonal_matrix;

namespace {

template<typename FloatType>
struct TestTraits {
    using other_float = float;
};

template<>
struct TestTraits<float> {
    using other_float = double;
};

} // namespace

TEMPLATE_LIST_TEST_CASE("block_diagonal_matrix", "",
                        types::floating_point_types) {
    using other_float = typename TestTraits<TestType>::other_float;
    Tensor square_matrix1(smooth_matrix_<TestType>());
    Tensor square_matrix2(smooth_matrix_<TestType>(3, 3));
    Tensor vector1(smooth_vector_<other_float>());
    Tensor vector2(smooth_vector_<TestType>());
    Tensor rectangular_matrix1(smooth_matrix_<TestType>(2, 3));
    std::vector<Tensor> inputs1{square_matrix1, square_matrix2};
    std::vector<Tensor> inputs2{square_matrix1, vector1};
    std::vector<Tensor> inputs3{square_matrix1, vector2};
    std::vector<Tensor> inputs4{square_matrix1, rectangular_matrix1};

    SECTION("All matrices are square") {
        shape::Smooth corr_shape{5, 5};
        layout::Physical corr_layout(corr_shape);
        auto allocator   = make_allocator<TestType>();
        auto corr_buffer = allocator.allocate(corr_layout);
        double counter1 = 1.0, counter2 = 1.0;
        for(std::size_t i = 0; i < 5; ++i) {
            for(std::size_t j = 0; j < 5; ++j) {
                if(i >= 2 and j >= 2)
                    corr_buffer->set_elem({i, j}, counter1++);
                else if(i < 2 and j < 2)
                    corr_buffer->set_elem({i, j}, counter2++);
                else
                    corr_buffer->set_elem({i, j}, 0.0);
            }
        }
        Tensor corr(corr_shape, std::move(corr_buffer));

        auto result = block_diagonal_matrix(inputs1);
        REQUIRE(result == corr);
    }

    SECTION("Input has different floating point types") {
        REQUIRE_THROWS(block_diagonal_matrix(inputs2));
    }

    SECTION("Input has non-matrix Tensor") {
        REQUIRE_THROWS(block_diagonal_matrix(inputs3));
    }

    SECTION("Input has reactangular matrix") {
        REQUIRE_THROWS(block_diagonal_matrix(inputs4));
    }
}
