// #include "../../buffer/make_pimpl.hpp"
// #include "../../test_tensor.hpp"
// #include <tensorwrapper/tensor/conversion/conversion.hpp>

// using namespace tensorwrapper;
// using namespace tensorwrapper::tensor;

// using scalar_tensor = TensorWrapper<field::Scalar>;
// using tot_tensor    = TensorWrapper<field::Tensor>;

// TEST_CASE("Dot") {
//     auto scalar_tensors = testing::get_tensors<field::Scalar>();
//     auto tot_tensors    = testing::get_tensors<field::Tensor>();

//     SECTION("Scalar Tensor") {
//         auto [vec_b, mat_b, t3d_b] = testing::make_pimpl<field::Scalar>();

//         to_ta_distarrayd_t converter;
//         using buffer_t = typename
//         to_ta_distarrayd_t::buffer_t<field::Scalar>;

//         buffer_t vec_buffer(std::move(vec_b));
//         buffer_t mat_buffer(std::move(mat_b));
//         buffer_t t3d_buffer(std::move(t3d_b));

//         SECTION("Vector") {
//             auto vec = scalar_tensors.at("vector");

//             scalar_tensor wrapped(vec);
//             const scalar_tensor const_wrapped(vec);

//             auto vec_ta = converter.convert(vec_buffer);

//             double vec_corr   = vec_ta("i").dot(vec_ta("i"));
//             auto vec_product  = dot(wrapped("i"), wrapped("i"));
//             auto const_first  = dot(const_wrapped("i"), wrapped("i"));
//             auto const_second = dot(wrapped("i"), const_wrapped("i"));

//             REQUIRE(vec_product == vec_corr);
//             REQUIRE(const_first == vec_corr);
//             REQUIRE(const_second == vec_corr);
//         }

//         SECTION("Matrix") {
//             auto mat = scalar_tensors.at("matrix");

//             scalar_tensor wrapped(mat);
//             auto mat_ta = converter.convert(mat_buffer);

//             double mat_corr  = mat_ta("i,j").dot(mat_ta("i,j"));
//             auto mat_product = dot(wrapped("i,j"), wrapped("i,j"));

//             REQUIRE(mat_product == mat_corr);
//         }

//         SECTION("Tensor") {
//             auto ten = scalar_tensors.at("tensor");

//             scalar_tensor wrapped(ten);
//             auto t3d_ta = converter.convert(t3d_buffer);

//             double ten_corr  = t3d_ta("i,j,k").dot(t3d_ta("i,j,k"));
//             auto ten_product = dot(wrapped("i,j,k"), wrapped("i,j,k"));

//             REQUIRE(ten_product == ten_corr);
//         }
//     }

//     SECTION("Tensor of Tensors") {
//         auto [vov_b, vom_b, mov_b] = testing::make_pimpl<field::Tensor>();

//         to_ta_totd_t converter;
//         using buffer_t = typename to_ta_totd_t::buffer_t<field::Tensor>;
//         buffer_t vov_buffer(std::move(vov_b));
//         buffer_t vom_buffer(std::move(vom_b));
//         buffer_t mov_buffer(std::move(mov_b));

//         SECTION("Vector of Vectors") {
//             auto vov = tot_tensors.at("vector-of-vectors");

//             tot_tensor wrapped(vov);
//             const tot_tensor const_wrapped(vov);

//             auto vov_ta = converter.convert(vov_buffer);

//             double vov_corr   = vov_ta("i;j").dot(vov_ta("i;j"));
//             auto vov_product  = dot(wrapped("i;j"), wrapped("i;j"));
//             auto const_first  = dot(const_wrapped("i;j"), wrapped("i;j"));
//             auto const_second = dot(wrapped("i;j"), const_wrapped("i;j"));

//             REQUIRE(vov_product == vov_corr);
//             REQUIRE(const_first == vov_corr);
//             REQUIRE(const_second == vov_corr);
//         }
//         SECTION("Matrix of Vectors") {
//             auto mov = tot_tensors.at("matrix-of-vectors");

//             tot_tensor wrapped(mov);
//             auto mov_ta = converter.convert(mov_buffer);

//             double mov_corr  = mov_ta("i,j;k").dot(mov_ta("i,j;k"));
//             auto mov_product = dot(wrapped("i,j;k"), wrapped("i,j;k"));

//             REQUIRE(mov_product == mov_corr);
//         }
//         SECTION("Vector of Matrices") {
//             auto vom = tot_tensors.at("vector-of-matrices");

//             tot_tensor wrapped(vom);
//             auto vom_ta = converter.convert(vom_buffer);

//             double vom_corr  = vom_ta("i;j,k").dot(vom_ta("i;j,k"));
//             auto vom_product = dot(wrapped("i;j,k"), wrapped("i;j,k"));

//             REQUIRE(vom_product == vom_corr);
//         }
//     }
// }
