#include "../../helpers.hpp"
#include <tensorwrapper/tensor/detail_/il_utils.hpp>

TEST_CASE("unwrap_il") {
    using dims_type    = std::deque<int>;
    using data_type    = std::vector<double>;
    using scalar_type  = double;
    using vector_type  = std::initializer_list<scalar_type>;
    using matrix_type  = std::initializer_list<vector_type>;
    using tensor3_type = std::initializer_list<matrix_type>;

    SECTION("scalar") {
        auto [dims, data] = unwrap_il(3.14);
        REQUIRE(dims == dims_type{});
        REQUIRE(data == data_type{3.14});
    }

    SECTION("vector") {
        auto [dims, data] = unwrap_il({3.14, 1.23});
        REQUIRE(dims == dims_type{2});
        REQUIRE(data == data_type{3.14, 1.23});
    }

    SECTION("matrix") {
        auto [dims, data] = unwrap_il(matrix_type{{3.14}, {1.23}});
        REQUIRE(dims == dims_type{2, 1});
        REQUIRE(data.size() == 2);
        REQUIRE(data == data_type{3.14, 1.23});

        // Jagged not supported yet
        using error_t = std::runtime_error;
        REQUIRE_THROWS_AS(unwrap_il(matrix_type{{3.14}, {}}), error_t);
    }

    SECTION("Rank 3 tensor") {
        auto [dims, data] =
          unwrap_il(tensor3_type{{{3.14}, {1.23}}, {{2.34}, {3.45}}});
        REQUIRE(dims == dims_type{2, 2, 1});
        REQUIRE(data == data_type{3.14, 1.23, 2.34, 3.45});
    }
}