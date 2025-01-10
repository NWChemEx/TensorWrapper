#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/contraction.hpp>
#include <tensorwrapper/buffer/eigen.hpp>

using namespace tensorwrapper;

TEST_CASE("runtime contraction infrastructure") {
    using float_type = float;
    using mode_type  = unsigned short;
    using pair_type  = std::pair<mode_type, mode_type>;
    // using mode_array = std::vector<pair_type>;

    // Inputs
    auto scalar  = testing::eigen_scalar<float_type>();
    auto vector  = testing::eigen_vector<float_type>();
    auto vector2 = testing::eigen_vector<float_type>(2);
    auto matrix  = testing::eigen_matrix<float_type>();

    // Buffers
    auto buffer1 = testing::eigen_vector<float_type>(2);
    auto buffer2 = testing::eigen_matrix<float_type>();

    pair_type p00{0, 0};
    pair_type p11{1, 1};

    auto scalar_corr      = testing::eigen_scalar<float>();
    scalar_corr.value()() = 30.0;

    auto vector_corr       = testing::eigen_vector<float>(2);
    vector_corr.value()(0) = 3.0;
    vector_corr.value()(1) = 4.0;

    auto matrix_corr          = testing::eigen_matrix<float>(2, 2);
    matrix_corr.value()(0, 0) = 10.0;
    matrix_corr.value()(0, 1) = 14.0;
    matrix_corr.value()(1, 0) = 14.0;
    matrix_corr.value()(1, 1) = 20.0;

    SECTION("contraction") {
        SECTION("vector with vector") {
            std::array modes{p00};
            auto& rv = buffer::contraction(scalar, vector, vector, modes);
            REQUIRE(scalar_corr.are_equal(rv));
        }

        SECTION("ij,ij->") {
            std::array modes{p00, p11};
            auto& rv = buffer::contraction(scalar, matrix, matrix, modes);
            REQUIRE(scalar_corr.are_equal(rv));
        }

        SECTION("ik,kj->ij") {
            std::array modes{p00};
            auto& rv = buffer::contraction(buffer2, matrix, matrix, modes);
            REQUIRE(matrix_corr.are_equal(rv));
        }

        SECTION("ij,i->j") {
            std::array modes{p00};
            auto& rv = buffer::contraction(buffer1, matrix, vector2, modes);
            REQUIRE(vector_corr.are_equal(rv));
        }
    }
}