#include "tensorwrapper/tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

using namespace tensorwrapper::tensor;

using ta_type     = TA::TSpArrayD;
using tensor_type = ScalarTensorWrapper;
using vector_il   = TA::detail::vector_il<double>;
using matrix_il   = TA::detail::matrix_il<double>;

namespace {

// Correct overlap of redundant_PAOs_corr
constexpr matrix_il redundant_PAO_overlap_corr{
  vector_il{0.41629351, -0.34700249}, vector_il{-0.34700249, 0.41629351}};

// Correct redundant PAOs for C_data and S_data
constexpr matrix_il redundant_PAOs_corr{vector_il{0.381648, -0.618352},
                                        vector_il{-0.618352, 0.381648}};

// Correct normalized, non-redundant PAOs for C_data and S_data
constexpr matrix_il NRC_corr_data{
  vector_il{0.6358462574920218, -0.8093539841320376},
  vector_il{0.6358462574920218, 0.8093539841320376}};

constexpr matrix_il NRC_1_corr_data{vector_il{-0.8093539841320377},
                                    vector_il{0.8093539841320377}};

} // namespace

TEST_CASE("remove_redundancy(TensorWrapper)") {
    auto& world      = TA::get_default_world();
    auto CTilde_corr = detail_::ta_to_tw(ta_type(world, redundant_PAOs_corr));
    auto STilde_corr =
      detail_::ta_to_tw(ta_type(world, redundant_PAO_overlap_corr));

    SECTION("No redundancy") {
        auto NRC      = remove_redundancy(CTilde_corr, STilde_corr);
        auto NRC_corr = detail_::ta_to_tw(ta_type(world, NRC_corr_data));
        REQUIRE(allclose(NRC, NRC_corr));
    }

    SECTION("One redundancy") {
        auto NRC = remove_redundancy(CTilde_corr, STilde_corr, 0.1);
        // Note this differs from NRC_corr_data by the first column being 0
        auto NRC_corr = detail_::ta_to_tw(ta_type(world, NRC_1_corr_data));
        REQUIRE(allclose(NRC, NRC_corr));
    }
}

// TEST_CASE("sparse_remove_redundancy") {
//     using tile_type   = TA::TensorD;
//     using tensor_type = TA::DistArray<TA::Tensor<tile_type>>;
//     auto& world       = TA::get_default_world();
//     using tvector_il  = TA::detail::vector_il<tile_type>;

//     TA::Range r(std::vector<int>{2, 2});
//     tile_type S0(r, {0.41629351, -0.34700249, -0.34700249, 0.41629351});
//     tensor_type S(world, tvector_il{S0, S0});
//     tile_type C0(r, {0.381648, -0.618352, -0.618352, 0.381648});
//     tensor_type C(world, tvector_il{C0, C0});

//     SECTION("No redundancy") {
//         auto NRC = sparse_remove_redundancy(C, S);
//         tile_type corr_tile(r,
//                             vector_il{0.6358462574920218,
//                             -0.8093539841320376,
//                                       0.6358462574920218,
//                                       0.8093539841320376});
//         tensor_type corr(world, tvector_il{corr_tile, corr_tile});
//         REQUIRE(allclose_tot(NRC, corr, 2));
//     }

//     SECTION("One redundancy") {
//         auto NRC = sparse_remove_redundancy(C, S, 0.1);
//         tile_type corr_tile(
//           r, {0.0, -0.8093539841320376, 0.0, 0.8093539841320376});
//         tensor_type corr(world, tvector_il{corr_tile, corr_tile});
//         REQUIRE(allclose_tot(NRC, corr, 2));
//     }
// }
