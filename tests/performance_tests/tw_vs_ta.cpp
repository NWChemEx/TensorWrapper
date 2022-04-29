#include "performance_tests.hpp"
<<<<<<< HEAD
=======
#include "tensorwrapper/tensor/creation.hpp"
#include "tensorwrapper/tensor/detail_/operations/add_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/mult_op.hpp"
#include "tensorwrapper/tensor/detail_/operations/subt_op.hpp"
#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"
#include "tiledarray.h"
>>>>>>> 57394757e77d2d01e6e8db6c19f121d7b771b38d

using namespace performance_tests;

// generate two random dummy tensors for test
namespace {
std::initializer_list<double> l1{rand_d, rand_d, rand_d, rand_d};
std::initializer_list<double> l2{rand_d, rand_d, rand_d, rand_d};
} // namespace

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
        tensorwrapper::tensor::detail_::MultOp<std::initializer_list<double>,
                                               std::initializer_list<double>>
          add_tw;
        auto begin_tw = std::chrono::high_resolution_clock::now();
        auto res_tw   = add_tw.MultOp(l1, l2);
        std::cout << 'TensorWrapper running time: '
                  << begin_tw.time_since_epoch() << std::endl;
    }

    SECTION("Add") {
        TA::TArrayD lhs_tensor{world, l1};
        TA::TArrayD rhs_tensor{world, l2};
        TA::TArrayD res_ta;
        // TA session
        auto begin_ta = std::chrono::high_resolution_clock::now();
        res_ta("i,j") = lhs_tensor("i,k") + rhs_tensor("k, j");
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;

        // TW session
        tensorwrapper::tensor::detail_::AddOp<std::initializer_list<double>,
                                              std::initializer_list<double>>
<<<<<<< HEAD
          add_tw;
        auto begin_tw = std::chrono::high_resolution_clock::now();
        auto res_tw   = add_tw.AddOp(l1, l2);
        std::cout << 'TensorWrapper running time: '
                  << begin_tw.time_since_epoch() << std::endl;
=======
          res_tw(l1, l2);
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;
>>>>>>> 57394757e77d2d01e6e8db6c19f121d7b771b38d
    }

    SECTION("Substract") {
        TA::TArrayD lhs_tensor{world, l1};
        TA::TArrayD rhs_tensor{world, l2};
        TA::TArrayD res_ta;
        // TA session
        auto begin_ta = std::chrono::high_resolution_clock::now();
        res_ta("i,j") = lhs_tensor("i,k") - rhs_tensor("k, j");
        std::cout << 'TiledArray running time: ' << begin_ta.time_since_epoch()
                  << std::endl;

        // TW session
        tensorwrapper::tensor::detail_::SubtOp<std::initializer_list<double>,
                                               std::initializer_list<double>>
          sub_tw;
        auto begin_tw = std::chrono::high_resolution_clock::now();
        auto res_tw   = sub_tw.SubtOp(l1, l2);
        std::cout << 'TensorWrapper running time: '
                  << begin_tw.time_since_epoch() << std::endl;
    }
}
