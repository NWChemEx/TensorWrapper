#include "../../test_tensor.hpp"

using namespace tensorwrapper;
using namespace tensorwrapper::tensor;

TEST_CASE("Dot") {
    SECTION("Scalar Tensor") {
        using scalar_traits = backends::TiledArrayTraits<field::Scalar>;
        using scalar_tensor = typename scalar_traits::tensor_type<double>;

        auto vec = testing::get_tensors<scalar_tensor>().at("vector");
        auto mat = testing::get_tensors<scalar_tensor>().at("matrix");
        auto ten = testing::get_tensors<scalar_tensor>().at("tensor");

        ScalarTensorWrapper wrapped_vec(vec);
        ScalarTensorWrapper wrapped_mat(mat);
        ScalarTensorWrapper wrapped_ten(ten);

        double vec_corr = vec("i").dot(vec("i"));
        double mat_corr = mat("i,j").dot(mat("i,j"));
        double ten_corr = ten("i,j,k").dot(ten("i,j,k"));

        auto vec_product = dot(wrapped_vec("i"), wrapped_vec("i"));
        auto mat_product = dot(wrapped_mat("i,j"), wrapped_mat("i,j"));
        auto ten_product = dot(wrapped_ten("i,j,k"), wrapped_ten("i,j,k"));

        REQUIRE(vec_product == vec_corr);
        REQUIRE(mat_product == mat_corr);
        REQUIRE(ten_product == ten_corr);
    }

    SECTION("Tensor of Tensors") {
        using tot_traits = backends::TiledArrayTraits<field::Tensor>;
        using tot_tensor = typename tot_traits::tensor_type<double>;

        auto vov = testing::get_tensors<tot_tensor>().at("vector-of-vectors");
        auto mov = testing::get_tensors<tot_tensor>().at("matrix-of-vectors");
        auto vom = testing::get_tensors<tot_tensor>().at("vector-of-matrices");

        TensorOfTensorsWrapper wrapped_vov(vov);
        TensorOfTensorsWrapper wrapped_mov(mov);
        TensorOfTensorsWrapper wrapped_vom(vom);

        double vov_corr = vov("i;j").dot(vov("i;j"));
        double mov_corr = mov("i,j;k").dot(mov("i,j;k"));
        double vom_corr = vom("i;j,k").dot(vom("i;j,k"));

        auto vov_product = dot(wrapped_vov("i;j"), wrapped_vov("i;j"));
        auto mov_product = dot(wrapped_mov("i,j;k"), wrapped_mov("i,j;k"));
        auto vom_product = dot(wrapped_vom("i;j,k"), wrapped_vom("i;j,k"));

        REQUIRE(vov_product == vov_corr);
        REQUIRE(mov_product == mov_corr);
        REQUIRE(vom_product == vom_corr);
    }
}