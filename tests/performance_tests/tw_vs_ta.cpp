#include "performance_tests.hpp"
#include "tensorwrapper/tensor/creation.hpp"
#include "tensorwrapper/tensor/detail_/operations/mult_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/add_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/subt_op.hpp"
#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"
#include "tiledarray.h"

using namespace performance_tests;

// generate two random dummy tensors for test
// namespace {
//   std::initializer_list<double> lhs_data{rand_d, rand_d, rand_d, rand_d};
//   std::initializer_list<double> rhs_data{rand_d, rand_d, rand_d, rand_d};
// }

std::initializer_list<double> l1{rand_d, rand_d, rand_d, rand_d};
std::initializer_list<double> l2{rand_d, rand_d, rand_d, rand_d};

TEST_CASE("Performance test: TensorWrapper v.s. TiledArray") {
    // using tensor_t = tensorwrapper::tensor::ScalarTensorWrapper;
    // auto tensor = performance_tests::gen_tensors<TA::TSpArrayD>();

    SECTION("Multiplication") {
        // TA session
        TA::TArrayD lhs_tensor{world, l1};
        TA::TArrayD rhs_tensor{world, l2};
        // fill the argument arrays with data
        // lhs_t.fill_local(rand_d);
        // lhs_t.fill_local(rand_d);
        TA::TArrayD res_ta;
        // run multiplication
        auto begin_ta = std::chrono::high_resolution_clock::now();
        res_ta("i,j") = lhs_tensor("i,k") * rhs_tensor("k, j");
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;

        // TW session
        auto begin_ta = std::chrono::high_resolution_clock::now();
        tensorwrapper::tensor::detail_::MultOp<std::initializer_list<double>,
                                               std::initializer_list<double>>
          res_tw(l1, l2);
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;
    }

    SECTION("Add") {
        TA::TArrayD lhs_tensor{world, l1};
        TA::TArrayD rhs_tensor{world, l2};
        // fill the argument arrays with data
        // lhs_t.fill_local(rand_d);
        // lhs_t.fill_local(rand_d);
        TA::TArrayD res_ta;
        // run multiplication
        auto begin_ta = std::chrono::high_resolution_clock::now();
        res_ta("i,j") = lhs_tensor("i,k") + rhs_tensor("k, j");
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;

        // TW session
        auto begin_ta = std::chrono::high_resolution_clock::now();
        tensorwrapper::tensor::detail_::AddOp<std::initializer_list<double>,
                                               std::initializer_list<double>>
          res_tw(l1, l2);
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;
    }

    SECTION("Substract") {
        TA::TArrayD lhs_tensor{world, l1};
        TA::TArrayD rhs_tensor{world, l2};
        TA::TArrayD res_ta;
        // run substraction
        auto begin_ta = std::chrono::high_resolution_clock::now();
        res_ta("i,j") = lhs_tensor("i,k") -rhs_tensor("k, j");
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;

        // TW session
        auto begin_ta = std::chrono::high_resolution_clock::now();
        tensorwrapper::tensor::detail_::SubtOp<std::initializer_list<double>,
                                               std::initializer_list<double>>
          res_tw(l1, l2);
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;
    }
}

// int main(int argc, char** argv) {
//     // Initialize the parallel runtime
//     TA::World& world = TA::initialize(argc, argv);
//     // Construct a 2D tiled range structure that defines
//     // the tiling of an array. Each dimension contains
//     // 10 tiles.
//     TA::TiledRange trange = {{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
//                              {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}};
//     // Construct and fill the argument arrays with data
//     TA::TArrayD A(world, trange);
//     TA::TArrayD B(world, trange);
//     A.fill_local(3.0);
//     B.fill_local(2.0);
//     // Construct the (empty) result array.
//     TA::TArrayD C;
//     // Perform a distributed matrix subtraction
//     C("i,j") = A("i,k") - B("k,j");
//     // Tear down the parallel runtime.
//     TA::finalize();
//     return 0;
// }