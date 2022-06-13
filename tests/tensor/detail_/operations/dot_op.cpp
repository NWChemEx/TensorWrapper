// #include "../../test_tensor.hpp"

// using namespace tensorwrapper;
// using namespace tensorwrapper::tensor;

// using scalar_traits = backends::TiledArrayTraits<field::Scalar>;
// using scalar_tensor = typename scalar_traits::tensor_type<double>;
// using tot_traits    = backends::TiledArrayTraits<field::Tensor>;
// using tot_tensor    = typename tot_traits::tensor_type<double>;

// TEST_CASE("Dot") {
//     auto scalar_tensors = testing::get_tensors<scalar_tensor>();
//     auto tot_tensors    = testing::get_tensors<tot_tensor>();

//     SECTION("Scalar Tensor") {
//         SECTION("Vector") {
//             auto vec = scalar_tensors.at("vector");

//             scalar_tensor wrapped(vec);
//             const scalar_tensor const_wrapped(vec);

//             double vec_corr   = vec("i").dot(vec("i"));
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

//             double mat_corr  = mat("i,j").dot(mat("i,j"));
//             auto mat_product = dot(wrapped("i,j"), wrapped("i,j"));

//             REQUIRE(mat_product == mat_corr);
//         }

//         SECTION("Tensor") {
//             auto ten = scalar_tensors.at("tensor");

//             scalar_tensor wrapped(ten);

//             double ten_corr  = ten("i,j,k").dot(ten("i,j,k"));
//             auto ten_product = dot(wrapped("i,j,k"), wrapped("i,j,k"));

//             REQUIRE(ten_product == ten_corr);
//         }
//     }

//     SECTION("Tensor of Tensors") {
//         SECTION("Vector of Vectors") {
//             auto vov = tot_tensors.at("vector-of-vectors");

//             tot_tensor wrapped(vov);
//             const tot_tensor const_wrapped(vov);

//             double vov_corr   = vov("i;j").dot(vov("i;j"));
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

//             double mov_corr  = mov("i,j;k").dot(mov("i,j;k"));
//             auto mov_product = dot(wrapped("i,j;k"), wrapped("i,j;k"));

//             REQUIRE(mov_product == mov_corr);
//         }
//         SECTION("Vector of Matrices") {
//             auto vom = tot_tensors.at("vector-of-matrices");

//             tot_tensor wrapped(vom);

//             double vom_corr  = vom("i;j,k").dot(vom("i;j,k"));
//             auto vom_product = dot(wrapped("i;j,k"), wrapped("i;j,k"));

//             REQUIRE(vom_product == vom_corr);
//         }
//     }
// }
