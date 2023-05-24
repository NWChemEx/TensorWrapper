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

#include "tensorwrapper/tensor/creation.hpp"
#include "tensorwrapper/tensor/tensor_wrapper.hpp"
#include "test_tensor.hpp"
#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/allocator/allocator.hpp>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("concatenate(Tensor)") {
    using tensor_t = tensorwrapper::tensor::ScalarTensorWrapper;
    auto tensors   = testing::get_tensors<field::Scalar>();
    auto& world    = TA::get_default_world();

    auto v = tensors.at("vector");
    auto m = tensors.at("matrix");

    using v_il = TA::detail::vector_il<double>;
    using m_il = TA::detail::matrix_il<double>;
    using t_il = TA::detail::tensor3_il<double>;

    SECTION("Vectors") {
        TA::TSpArrayD corr_ta(world, v_il{1.0, 2.0, 3.0, 1.0, 2.0, 3.0});
        auto corr = detail_::ta_to_tw(corr_ta);
        auto rv   = concatenate(v, v, 0);
        REQUIRE(rv == corr);
    }

    SECTION("Matrix") {
        SECTION("Mode 0") {
            m_il il{v_il{1.0, 2.0}, v_il{3.0, 4.0}, v_il{1.0, 2.0},
                    v_il{3.0, 4.0}};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = concatenate(m, m, 0);
            REQUIRE(rv == corr);
        }
        SECTION("Mode 1") {
            m_il il{v_il{1.0, 2.0, 1.0, 2.0}, v_il{3.0, 4.0, 3.0, 4.0}};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = concatenate(m, m, 1);
            REQUIRE(rv == corr);
        }
    }

    SECTION("Throws if attempting to concatenate different ranks") {
        REQUIRE_THROWS_AS(concatenate(v, m, 1), std::runtime_error);
    }

    SECTION("Throws if dimension is not valid") {
        REQUIRE_THROWS_AS(concatenate(v, v, 1), std::runtime_error);
    }

    SECTION("Throws if shapes are not compatible") {
        m_il il{v_il{1.0}};
        TA::TSpArrayD other_m_ta(world, il);
        auto other_m = detail_::ta_to_tw(other_m_ta);
        REQUIRE_THROWS_AS(concatenate(m, other_m, 0), std::runtime_error);
    }
}

TEST_CASE("concatenate(ToT)") {
    TensorOfTensorsWrapper A, B;
    REQUIRE_THROWS_AS(concatenate(A, B, 0), std::runtime_error);
}

TEST_CASE("diagonal_tensor_wrapper") {
    using tensor_t  = tensorwrapper::tensor::ScalarTensorWrapper;
    using scalar_t  = tensorwrapper::tensor::field::Scalar;
    using shape_t   = tensorwrapper::tensor::Shape<scalar_t>;
    using extents_t = typename shape_t::extents_type;
    using v_il      = TA::detail::vector_il<double>;
    using m_il      = TA::detail::matrix_il<double>;
    using t_il      = TA::detail::tensor3_il<double>;

    auto& world = TA::get_default_world();
    auto p      = tensorwrapper::tensor::default_allocator<scalar_t>();

    v_il v1{2.0, 2.0, 2.0};
    v_il v2{2.0, 0.0};
    v_il v3{0.0, 2.0};
    v_il v4{0.0, 0.0};
    v_il v5{1.0, 2.0, 3.0};
    v_il v6{1.0, 0.0};

    m_il m1{v2, v3};
    m_il m2{v2, v4};
    m_il m3{v4, v3};
    m_il m4{v6, v3};
    m_il m5{v6, v4};
    m_il m6{v4, v3};
    m_il m7{v6, v3, v4};

    t_il t1{m2, m3};
    t_il t2{m5, m6};

    SECTION("Single Diagonal Value") {
        SECTION("1D") {
            extents_t extents{3};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, v1));
            auto rv   = diagonal_tensor_wrapper(2.0, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("2D") {
            extents_t extents{2, 2};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, m1));
            auto rv   = diagonal_tensor_wrapper(2.0, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("3D") {
            extents_t extents{2, 2, 2};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, t1));
            auto rv   = diagonal_tensor_wrapper(2.0, *p, shape);
            REQUIRE(rv == corr);
        }
    }

    SECTION("Multiple Diagonal Values") {
        std::vector<double> two_vals   = {1.0, 2.0};
        std::vector<double> three_vals = {1.0, 2.0, 3.0};

        SECTION("1D") {
            extents_t extents{3};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, v5));
            auto rv   = diagonal_tensor_wrapper(three_vals, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("2D") {
            extents_t extents{2, 2};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, m4));
            auto rv   = diagonal_tensor_wrapper(two_vals, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("3D") {
            extents_t extents{2, 2, 2};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, t2));
            auto rv   = diagonal_tensor_wrapper(two_vals, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("Rectangular") {
            extents_t extents{3, 2};
            shape_t shape{extents};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, m7));
            auto rv   = diagonal_tensor_wrapper(two_vals, *p, shape);
            REQUIRE(rv == corr);
        }

        SECTION("Too few values") {
            extents_t extents{3};
            shape_t shape{extents};
            REQUIRE_THROWS_AS(diagonal_tensor_wrapper(two_vals, *p, shape),
                              std::runtime_error);
        }
    }

    SECTION("Block Diagonal Values") {
        std::vector<std::vector<double>> test_vals{{1.0}, {2.0, 3.0, 4.0, 5.0}};

        SECTION("1D") {
            extents_t extents{3};
            shape_t shape{extents};
            auto rv = diagonal_tensor_wrapper(test_vals, *p, shape);
            tensor_t corr({1.0, 2.0, 3.0});
            REQUIRE(rv == corr);
        }

        SECTION("2D") {
            extents_t extents{3, 3};
            shape_t shape{extents};
            auto rv = diagonal_tensor_wrapper(test_vals, *p, shape);
            tensor_t corr({{1.0, 0.0, 0.0}, {0.0, 2.0, 3.0}, {0.0, 4.0, 5.0}});
            REQUIRE(rv == corr);
        }

        SECTION("3D") {
            std::vector<std::vector<double>> test_vals2{
              {1.0}, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}};
            extents_t extents{3, 3, 3};
            shape_t shape{extents};
            auto rv = diagonal_tensor_wrapper(test_vals2, *p, shape);
            tensor_t corr(
              {{{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
               {{0.0, 0.0, 0.0}, {0.0, 2.0, 3.0}, {0.0, 4.0, 5.0}},
               {{0.0, 0.0, 0.0}, {0.0, 6.0, 7.0}, {0.0, 8.0, 9.0}}});
            REQUIRE(rv == corr);
        }

        SECTION("Rectangular") {
            extents_t extents{3, 2};
            shape_t shape{extents};
            auto rv = diagonal_tensor_wrapper(test_vals, *p, shape);
            tensor_t corr({{1.0, 0.0}, {0.0, 2.0}, {0.0, 4.0}});
            REQUIRE(rv == corr);
        }

        SECTION("Too few values") {
            extents_t extents{4, 4};
            shape_t shape{extents};
            REQUIRE_THROWS_AS(diagonal_tensor_wrapper(test_vals, *p, shape),
                              std::runtime_error);
        }

        SECTION("Block not square") {
            extents_t extents{3, 1, 1};
            shape_t shape{extents};
            REQUIRE_THROWS_AS(diagonal_tensor_wrapper(test_vals, *p, shape),
                              std::runtime_error);
        }
    }
}

TEST_CASE("stack_tensors") {
    using tensor_t = tensorwrapper::tensor::ScalarTensorWrapper;
    using scalar_t = tensorwrapper::tensor::field::Scalar;
    auto tensors   = testing::get_tensors<scalar_t>();
    auto& world    = TA::get_default_world();

    using v_il = TA::detail::vector_il<double>;
    using m_il = TA::detail::matrix_il<double>;
    using t_il = TA::detail::tensor3_il<double>;

    auto v = tensors.at("vector");
    auto m = tensors.at("matrix");

    SECTION("1D to 2D") {
        v_il il1{1.0, 2.0, 3.0};

        SECTION("One vector") {
            m_il il{il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({v});
            REQUIRE(rv == corr);
        }
        SECTION("Two vectors") {
            m_il il{il1, il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({v, v});
            REQUIRE(rv == corr);
        }
        SECTION("Three vectors") {
            m_il il{il1, il1, il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({v, v, v});
            REQUIRE(rv == corr);
        }
    }

    SECTION("2D to 3D") {
        m_il il1{v_il{1.0, 2.0}, v_il{3.0, 4.0}};

        SECTION("One matrix") {
            t_il il{il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({m});
            REQUIRE(rv == corr);
        }
        SECTION("Two matrices") {
            t_il il{il1, il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({m, m});
            REQUIRE(rv == corr);
        }
        SECTION("Three matrices") {
            t_il il{il1, il1, il1};
            auto corr = detail_::ta_to_tw(TA::TSpArrayD(world, il));
            auto rv   = stack_tensors({m, m, m});
            REQUIRE(rv == corr);
        }
    }

    SECTION("Throws if shapes are not compatible") {
        REQUIRE_THROWS_AS(stack_tensors({v, m}), std::runtime_error);
    }
}

TEST_CASE("Eigen conversions") {
    using tensor_t = tensorwrapper::tensor::ScalarTensorWrapper;
    using scalar_t = tensorwrapper::tensor::field::Scalar;
    using shape_t  = typename tensor_t::shape_type;
    using tiling_t = typename shape_t::tiling_type;
    auto twrapper  = testing::get_tensors<scalar_t>().at("matrix");

    Eigen::MatrixXd eigen_m(2, 2);
    eigen_m(0, 0) = 1.0;
    eigen_m(0, 1) = 2.0;
    eigen_m(1, 0) = 3.0;
    eigen_m(1, 1) = 4.0;

    SECTION("tensor_wrapper_to_eigen") {
        auto rv = tensor_wrapper_to_eigen(twrapper);
        REQUIRE(rv == eigen_m);
    }

    SECTION("eigen_to_tensor_wrapper (default shape)") {
        auto rv = eigen_to_tensor_wrapper(eigen_m);
        REQUIRE(rv == twrapper);
        tiling_t one_big_tile{{0,2},{0,2}};
        shape_t one_big_tile_shape(one_big_tile);
        REQUIRE(rv.shape() == one_big_tile_shape);
    }

    SECTION("eigen_to_tensor_wrapper (specified shape)") {
        tiling_t single_element_row_tile{{0,1,2},{0,2}};
        shape_t single_element_row_tile_shape(single_element_row_tile);
        auto rv = eigen_to_tensor_wrapper(eigen_m, single_element_row_tile_shape);
        REQUIRE(rv.shape() == single_element_row_tile_shape);
#if 1
        // Convert resulting tensor back to eigen
        auto eigen_rv = tensor_wrapper_to_eigen(rv);
        REQUIRE(eigen_rv == eigen_m);
#else
        twrapper.reshape(single_element_row_tile_shape.clone());
        REQUIRE(rv == twrapper);
#endif
    }
}
