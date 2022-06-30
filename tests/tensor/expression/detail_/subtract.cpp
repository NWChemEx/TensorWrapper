#include "../../test_tensor.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

using namespace tensorwrapper::tensor;

/* Testing Strategy
 *
 * - For classes derived from NNary we only need to test that tensor_ is
 *   implemented correctly (ctor, clone_, and are_equal_ are tested in
 *   nnary.cpp)
 * - These calls ultimately call Buffer::subtract, which is already known to
 * work. Hence we only need to check that the labels and the tensors correctly
 * get mapped to that call. The easiest way to test this is to evaluate the
 *   operation with different tensors and label combinations and ensure we get
 *   the correct answer.
 */

TEST_CASE("Subtract<field::Scalar>") {
    using field_type  = field::Scalar;
    using tensor_type = TensorWrapper<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    tensor_type b{{5.0, 6.0}, {7.0, 8.0}};

    auto amb  = a("i,j") - b("i,j");
    auto ambt = a("i,j") - b("j,i");

    SECTION("tensor_") {
        SECTION("c = a - b") {
            // C starts empty, so we know all the buffers get mapped correctly
            tensor_type c, corr{{-4.0, -4.0}, {-4.0, -4.0}};
            c = amb.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("c = a - bt") {
            // Checks that b's labels get mapped to b
            tensor_type c, corr{{-4.0, -5.0}, {-3.0, -4.0}};
            c = ambt.tensor("i,j", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
        SECTION("c = at - b") {
            // Checks that a's labels get mapped to a
            tensor_type c, corr{{-4.0, -3.0}, {-5.0, -4.0}};
            c = ambt.tensor("j,i", corr.shape(), corr.allocator());
            REQUIRE(allclose(c, corr));
        }
    }
}
