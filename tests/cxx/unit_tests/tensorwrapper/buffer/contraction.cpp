#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/contraction.hpp>
#include <tensorwrapper/buffer/eigen.hpp>

using namespace tensorwrapper;

TEST_CASE("contraction") {
    using float_type = float;
    using pair_type  = std::pair<int, int>;
    // using mode_array = std::vector<pair_type>;

    SECTION("vector with vector") {
        auto scalar = testing::eigen_scalar<float_type>();
        auto vector = testing::eigen_vector<float_type>();

        std::array modes{pair_type{0, 0}};
        auto& rv       = buffer::contraction(scalar, vector, vector, modes);
        auto corr      = testing::eigen_scalar<float>();
        corr.value()() = 30.0; // 0 + 1 + 4 + 9 + 16
        REQUIRE(corr.are_equal(rv));
    }

    SECTION("matrix with matrix") {
        auto scalar = testing::eigen_scalar<float_type>();
        auto vector = testing::eigen_vector<float_type>();
        auto matrix = testing::eigen_matrix<float_type>();

        SECTION("Down to scalar") {
            std::array modes{pair_type{0, 0}, pair_type{1, 1}};
            auto& rv       = buffer::contraction(scalar, matrix, matrix, modes);
            auto corr      = testing::eigen_scalar<float>();
            corr.value()() = 30.0; // 0 + 1 + 4 + 9 + 16
            std::cout << rv << std::endl;
            REQUIRE(corr.are_equal(rv));
        }

        SECTION("Down to vector") {
            std::array modes{pair_type{0, 0}};
            auto& rv  = buffer::contraction(vector, matrix, matrix, modes);
            auto corr = testing::eigen_vector<float>(2);
            corr.value()(0) = 30.0; // 0 + 1 + 4 + 9 + 16
            corr.value()(1) = 30.0;
            std::cout << rv << std::endl;
            REQUIRE(corr.are_equal(rv));
        }
    }
}