#include "../../test_tensor.hpp"
#include <tensorwrapper/tensor/conversion/conversion.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

using namespace tensorwrapper::tensor;

/* Testing Strategy
 *
 * In theory we can unit test the Add class directly by:
 *
 * - making labeled view instances
 * - converting them to expression instances
 * - creating a new Add instance with said expression instances
 * - calling eval on the Add instance
 * - examining the results of the resulting tensor
 *
 */

TEST_CASE("Add<field::Scalar>") {
    using field_type     = field::Scalar;
    using tensor_type    = TensorWrapper<field_type>;
    using converter_type = to_ta_distarrayd_t;
    using ta_type        = typename converter_type::output_t;

    converter_type converter;
    auto tensors = testing::get_tensors<field_type>();

    auto& mat    = tensors.at("matrix");
    auto& t3     = tensors.at("tensor");
    auto& ta_mat = converter.convert(mat.buffer());
    auto& ta_t3  = converter.convert(t3.buffer());

    SECTION("vector + vector") {
        auto& tw = tensors.at("vector");
        auto& ta = converter.convert(tw.buffer());

        tensor_type tw_rv;
        tw_rv("i") = tw("i") + tw("i");

        ta_type ta_rv;
        ta_rv("i") = ta("i") + ta("i");

        std::cout << tw_rv << std::endl;
        std::cout << ta_rv << std::endl;
    }
}
