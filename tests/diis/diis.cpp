#include <catch2/catch.hpp>
#include <tensorwrapper/detail_/hashing.hpp>
#include <tensorwrapper/diis/diis.hpp>
#include <tensorwrapper/tensor/allclose.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

using diis_type   = tensorwrapper::diis::DIIS;
using tensor_type = tensorwrapper::tensor::ScalarTensorWrapper;

/// For making input values
using ta_type   = TA::TSpArrayD;
using vector_il = TA::detail::vector_il<double>;
using matrix_il = TA::detail::matrix_il<double>;

namespace {

/// Relatively nonsensical input values and the outputs associated with them.
/// i1 is the value of both the first input and output
constexpr matrix_il i1{vector_il{1, 2}, vector_il{3, 4}};

constexpr matrix_il i2{vector_il{6, 5}, vector_il{8, 7}};

constexpr matrix_il i3{vector_il{12, 11}, vector_il{10, 9}};

constexpr matrix_il o2{vector_il{12, 8.6}, vector_il{14, 10.6}};

constexpr matrix_il o3{vector_il{15.35294118, 14.35294118},
                       vector_il{11.11764706, 10.11764706}};

} // namespace

TEST_CASE("DIIS") {
    auto& world = TA::get_default_world();

    // Inputs and expected values for extrapolation
    tensor_type input1(ta_type(world, i1));
    tensor_type input2(ta_type(world, i2));
    tensor_type input3(ta_type(world, i3));
    tensor_type corr_output1(ta_type(world, i1));
    tensor_type corr_output2(ta_type(world, o2));
    tensor_type corr_output3(ta_type(world, o3));

    // Different DIIS instances
    auto diis_default = diis_type();
    auto diis_max_2   = diis_type(2);
    auto diis_used    = diis_type();
    auto temp         = diis_used.extrapolate(input1, input3);

    SECTION("CTors") {
        SECTION("Typedefs") {
            SECTION("size_type") {
                using corr_t = std::size_t;
                using the_t  = diis_type::size_type;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }
            SECTION("tensor_type") {
                using corr_t = tensor_type;
                using the_t  = diis_type::tensor_type;
                STATIC_REQUIRE(std::is_same_v<corr_t, the_t>);
            }
        }
        SECTION("Default") { REQUIRE(diis_default == diis_type()); }
        SECTION("With Value") { REQUIRE(diis_max_2 == diis_type(2)); }
    }

    SECTION("extrapolate") {
        // Call extrapolate enough to require removing an old value
        auto diis    = diis_type(2);
        auto output1 = diis.extrapolate(input1, input3);
        auto output2 = diis.extrapolate(input2, input2);
        auto output3 = diis.extrapolate(input3, input1);

        using tensorwrapper::tensor::allclose;
        REQUIRE(allclose(output1, corr_output1));
        REQUIRE(allclose(output2, corr_output2));
        REQUIRE(allclose(output3, corr_output3));
    }

    SECTION("comparisons") {
        SECTION("Max vec not equal") { REQUIRE(diis_max_2 != diis_default); }
        SECTION("Recorded values different") {
            REQUIRE(diis_default != diis_used);
        }
    }

    SECTION("hash") {
        using tensorwrapper::detail_::hash_objects;
        auto h_default    = hash_objects(diis_default);
        auto h_with_value = hash_objects(diis_max_2);
        auto h_used       = hash_objects(diis_used);

        REQUIRE(h_default == hash_objects(diis_type()));
        REQUIRE(h_with_value == hash_objects(diis_type(2)));
        REQUIRE(h_default != h_with_value);
        REQUIRE(h_default != h_used);
    }
}