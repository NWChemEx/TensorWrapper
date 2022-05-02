/*
 * Functions, types, and includes common to the performance test.
 */
#pragma once
#include "performance_tests.hpp"
// #include "tensorwrapper/ta_helpers/tensor.hpp"
// #include "tensorwrapper/tensor/creation.hpp"
#include "tensorwrapper/tensor/detail_/operations/add_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/mult_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/subt_op.hpp"
#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"
#include "tiledarray.h"
#include < catch2 / catch.hpp>
#include <chrono>
#include <iostream>
#include <random>

namespace performance_tests {
// generate random dummpy tensors
constexpr double MIN = 1.0;
constexpr double MAX = 100.0;
std::random_device rd;
std::mt19937 eng(rd());
std::uniform_int_distribution<int> distr(MIN, MAX);
using rand_d = distr(eng)

  template<typename TensorType>
  auto gen_tensors() {
    auto& world = TA::get_default_world();
    std::map<std::string, TensorType> res;
    // use std namespace instead of TA
    using vector_il = std::initializer_list<double>;
    using matrix_il = std::initializer_list<vector_il<double>>;
    using tensor_il = std::initializer_list<matrix_il<double>>;

    // not tensor of tensor
    if(constexpr(!tensorwrapper::tensor::TensorTraits<TensorType>::is_tot)) {
        res["vector"] = TensorType(world, vector_il{rand_d, rand_d, rand_d});
        res["matrix"] = TensorType(world, matrix_il{vector_il{rand_d, rand_d},
                                                    vector_il{rand_d, rand_d}});
      res["tensor"] = TensorType(
        world,
        tensor_il{matrix_il{vector_il{rand_d, rand_d}, vector_ilrand_d, rand_d}},
                  matrix_il{
            vector_il{rand_d, rand_d}, vector_il { rand_d, rand_d }}
    });
}
else {
    using outer_tile = typename TensorType::value_type;
    using inner_tile = typename outer_tile::value_type;
    using dvector_il = vector_il;
    using vector_il  = std::initializer_list<inner_tile>;
    using matrix_il  = std::initializer_list<vector_il<inner_tile>>;

    inner_tile v0(TA::Range({2}), {rand_d, rand_d});
    inner_tile v1(TA::Range({2}), {rand_d, rand_d});
    inner_tile v2(TA::Range({2}), {rand_d, rand_d});
    inner_tile v3(TA::Range({2}), {rand_d, rand_d});
    inner_tile mat0(TA::Range({2, 2}),
                    dvector_il{rand_d, rand_d, rand_d, rand_d});
    inner_tile mat1(TA::Range({2, 2}),
                    dvector_il{rand_d, rand_d, rand_d, rand_d});
    rv["vector-of-vectors"] = TensorType(world, vector_il{v0, v1});
    rv["matrix-of-vectors"] =
      TensorType(world, matrix_il{vector_il{v0, v1}, vector_il{v2, v3}});
    rv["vector-of-matrices"] = TensorType(world, vector_il{mat0, mat1});
}
return rv;
}
} // namespace performance_tests
