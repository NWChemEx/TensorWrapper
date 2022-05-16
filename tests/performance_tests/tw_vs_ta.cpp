// #include "tensorwrapper/tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <chrono>
#include <iostream>
#include <tiledarray.h>
using namespace std::chrono;
#include "../tensor/test_tensor.hpp"

using namespace tensorwrapper;
using namespace tensorwrapper::tensor;
using scalar_traits  = backends::TiledArrayTraits<field::Scalar>;
using scalar_variant = typename scalar_traits::variant_type;
using scalar_tensor  = typename scalar_traits::tensor_type<double>;

TEST_CASE("Performance test: TensorWrapper v.s. TiledArray") {
    SECTION("mult") {
        auto lhs = testing::get_tensors<scalar_tensor>().at("vector");
        auto rhs = testing::get_tensors<scalar_tensor>().at("vector");
        // madness::initialized();
        auto& world = TA::get_default_world();
        scalar_tensor res_ta;
        world.gop.fence();
        auto start_ta = high_resolution_clock::now();
        res_ta("i,j") = lhs("i") * rhs("j");
        world.gop.fence();
        auto stop_ta     = high_resolution_clock::now();
        auto duration_ta = duration_cast<microseconds>(stop_ta - start_ta);

        std::cout << "performance test:" << std::endl;
        std::cout << "Time taken by operation(TA): " << duration_ta.count()
                  << " microseconds" << std::endl;

        ScalarTensorWrapper wrapped_lhs(lhs);
        ScalarTensorWrapper wrapped_rhs(rhs);
        ScalarTensorWrapper result(scalar_tensor{});
        world.gop.fence();
        auto start_tw  = high_resolution_clock::now();
        result("i, j") = wrapped_lhs("i") * wrapped_rhs("j");
        world.gop.fence();
        auto stop_tw     = high_resolution_clock::now();
        auto duration_tw = duration_cast<microseconds>(stop_tw - start_tw);

        std::cout << "Time taken by operation(TW): " << duration_tw.count()
                  << " microseconds" << std::endl;
    }
}
