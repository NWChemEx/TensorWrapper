// SECTION("permute_assignment_") {
//         SECTION("scalar") {
//             auto scalar2      = testing::eigen_scalar<TestType>();
//             scalar2.value()() = 42.0;

//             auto s        = scalar("");
//             auto pscalar2 = &(scalar2.permute_assignment("", s));
//             REQUIRE(pscalar2 == &scalar2);
//             REQUIRE(scalar2 == scalar);
//         }

//         SECTION("vector") {
//             auto vector2 = testing::eigen_vector<TestType>();

//             auto vi       = vector("i");
//             auto pvector2 = &(vector2.permute_assignment("i", vi));

//             REQUIRE(pvector2 == &vector2);
//             REQUIRE(vector2 == vector);
//         }

//         SECTION("matrix : no permutation") {
//             auto matrix2 = testing::eigen_matrix<TestType>();

//             auto mij      = matrix("i,j");
//             auto pmatrix2 = &(matrix2.permute_assignment("i,j", mij));

//             REQUIRE(pmatrix2 == &matrix2);
//             REQUIRE(matrix2 == matrix);
//         }

//         SECTION("matrix: permutation") {
//             auto matrix2 = testing::eigen_matrix<TestType>();
//             auto p       = &(matrix2.permute_assignment("j,i",
//             matrix("i,j")));

//             auto corr          = testing::eigen_matrix<TestType>(3, 2);
//             corr.value()(0, 0) = 10.0;
//             corr.value()(1, 0) = 20.0;
//             corr.value()(2, 0) = 30.0;
//             corr.value()(0, 1) = 40.0;
//             corr.value()(1, 1) = 50.0;
//             corr.value()(2, 1) = 60.0;
//             REQUIRE(p == &matrix2);
//             compare_eigen<TestType>(corr.value(), matrix2.value());
//         }
//     }