#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers;
using tensor_type = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;
using vector_il   = TA::detail::vector_il<double>;
using matrix_il   = TA::detail::matrix_il<double>;

TEST_CASE("allclose") {
    auto& world = TA::get_default_world();
    SECTION("identical") {
        tensor_type lhs(world,
                        matrix_il{vector_il{0.0, 1.1}, vector_il{2.2, 3.3}});
        tensor_type corr(world,
                         matrix_il{vector_il{0.0, 1.1}, vector_il{2.2, 3.3}});
        REQUIRE(allclose(lhs, corr));
    }

    SECTION("Absolute difference 0.1") {
        tensor_type lhs(world,
                        matrix_il{vector_il{0.0, 1.1}, vector_il{2.2, 3.3}});
        tensor_type corr(world,
                         matrix_il{vector_il{0.0, 1.2}, vector_il{2.2, 3.3}});

        SECTION("Close if atol == 0.1") {
            REQUIRE(allclose(lhs, corr, false, 0.0, 0.1));
        }

        SECTION("Not close if atol < 0.1") {
            REQUIRE_FALSE(allclose(lhs, corr, false, 0.0, 0.09));
        }
    }

    SECTION("Relative difference 0.1") {
        tensor_type lhs(world,
                        matrix_il{vector_il{0.0, 1.09}, vector_il{2.2, 3.3}});
        tensor_type corr(world,
                         matrix_il{vector_il{0.0, 1.2}, vector_il{2.2, 3.3}});

        SECTION("Close if rtol == 0.1") {
            REQUIRE(allclose(lhs, corr, false, 0.1));
        }

        SECTION("Not close if rtol < 0.1") {
            REQUIRE_FALSE(allclose(lhs, corr, false, 0.09));
        }
    }

    SECTION("Compare absolute values") {
        tensor_type lhs(world,
                        matrix_il{vector_il{0.0, -1.1}, vector_il{2.2, 3.3}});
        tensor_type corr(world,
                         matrix_il{vector_il{0.0, 1.1}, vector_il{-2.2, 3.3}});
        REQUIRE(allclose(lhs, corr, true));
        REQUIRE_FALSE(allclose(lhs, corr, false));
    }

    SECTION("Actual is Sparse Tensor with missing blocks") {
        TA::TiledRange trange{{0, 2}, {0, 2}};

        auto l   = [](TA::TensorD&, const TA::Range&) { return 0.0; };
        auto lhs = TA::make_array<tensor_type>(world, trange, l);

        SECTION("Close to explicitly zero tensor") {
            tensor_type corr(
              world, matrix_il{vector_il{0.0, 0.0}, vector_il{0.0, 0.0}});
            REQUIRE(allclose(lhs, corr));
        }

        SECTION("Not close to non-zero tensor") {
            tensor_type corr(
              world, matrix_il{vector_il{0.0, 1.1}, vector_il{2.2, 3.3}});
            REQUIRE_FALSE(allclose(lhs, corr));
        }
    }

    SECTION("Correct is Sparse Tensor with missing blocks") {
        TA::TiledRange trange{{0, 2}, {0, 2}};

        auto l    = [](TA::TensorD&, const TA::Range&) { return 0.0; };
        auto corr = TA::make_array<tensor_type>(world, trange, l);

        SECTION("Close to explicitly zero tensor") {
            tensor_type lhs(
              world, matrix_il{vector_il{0.0, 0.0}, vector_il{0.0, 0.0}});
            REQUIRE(allclose(lhs, corr));
        }

        SECTION("Not close to non-zero tensor") {
            tensor_type lhs(
              world, matrix_il{vector_il{0.0, 1.1}, vector_il{2.2, 3.3}});
            // TODO: Re-enable after TA #184 is resolved
            // REQUIRE_FALSE(allclose(lhs, corr));
        }
    }
}

TEST_CASE("Tensor Creation") {
    auto& world = TA::get_default_world();
    tensor_type matrix(world,
                       matrix_il{vector_il{1.0, 2.0}, vector_il{3.0, 4.0}});
    tensor_type corr(world,
                     matrix_il{vector_il{2.0, 4.0}, vector_il{6.0, 8.0}});

    SECTION("apply_elementwise") {
        auto doubled =
          apply_elementwise(matrix, [](const double& old) { return 2 * old; });
        REQUIRE(allclose(doubled, corr));
    }

    SECTION("apply_elementwise_inplace") {
        apply_elementwise_inplace(matrix, [](double& value) { value *= 2; });
        REQUIRE(allclose(matrix, corr));
    }

    SECTION("grag_diagonal") {
        auto diag = grab_diagonal(matrix);
        tensor_type corr_diag(world, vector_il{1.0, 4.0});
        REQUIRE(allclose(diag, corr_diag));
    }

    SECTION("array_from_vec") {
        std::vector<double> vec{1.0, 2.0, 3.0};
        tensor_type corr_vec(world, vector_il{1.0, 2.0, 3.0});
        auto& trange = corr_vec.trange().dim(0);
        auto ta_vec  = array_from_vec(vec, trange, world);
        REQUIRE(allclose(ta_vec, corr_vec));
    }
}

TEST_CASE("TiledRange1 Creation") {
    SECTION("Default init_offset") {
        auto corr_tr = TA::TiledRange1({0, 2, 4, 5});
        auto test_tr = make_1D_trange(5, 2);
        REQUIRE(test_tr == corr_tr);
    }

    SECTION("Non-default init_offset") {
        auto corr_tr = TA::TiledRange1({1, 3, 5});
        auto test_tr = make_1D_trange(5, 2, 1);
        REQUIRE(test_tr == corr_tr);
    }
}