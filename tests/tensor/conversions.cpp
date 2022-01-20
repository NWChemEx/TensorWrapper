#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"

using namespace tensorwrapper::tensor;

TEST_CASE("to_vector") {
    using tensor_type = tensorwrapper::tensor::ScalarTensorWrapper;
    auto tensors      = testing::get_tensors<TA::TSpArrayD>();

    SECTION("vector") {
        const tensor_type t(tensors.at("vector"));
        std::vector<double> corr{1.0, 2.0, 3.0};
        REQUIRE(to_vector(t) == corr);
    }

    SECTION("matrix") {
        const tensor_type t(tensors.at("matrix"));
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0};
        REQUIRE(to_vector(t) == corr);
    }

    SECTION("tensor") {
        const tensor_type t(tensors.at("tensor"));
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        REQUIRE(to_vector(t) == corr);
    }

    // The conversion previously had a bug that incorrectly computed the
    // flattened offset when the tensor had more than one tile. This test
    // ensures that bug doesn't come back
    SECTION("More than one tile") {
        using field_t          = field::Scalar;
        using single_element_t = SingleElementTiles<field_t>;
        auto palloc            = std::make_unique<single_element_t>();
        tensor_type t(tensors.at("matrix"), std::move(palloc));
        std::vector<double> corr{1.0, 2.0, 3.0, 4.0};
        REQUIRE(to_vector(t) == corr);
    }
}
