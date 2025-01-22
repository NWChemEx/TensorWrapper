/*
 * Copyright 2024 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace testing;

#ifdef ENABLE_SIGMA
using types2test = std::tuple<float, double, sigma::UFloat, sigma::UDouble>;
#else
using types2test = std::tuple<float, double>;
#endif

namespace {

template<typename FloatType, typename LHSType, typename RHSType>
void compare_eigen(const LHSType& lhs, const RHSType& rhs) {
    using r_type = Eigen::Tensor<FloatType, 0, Eigen::RowMajor>;
    auto d       = lhs - rhs;
    r_type r     = d.sum();
    // sigma's types are the only none fundamental options here currently
    // This may need to be adjusted if other custom numeric types are added
    if constexpr(std::is_fundamental_v<FloatType>) {
        REQUIRE_THAT(r() + 1.0, Catch::Matchers::WithinAbs(1.0, 1E-6));
    } else {
        REQUIRE_THAT(r().mean() + 1.0, Catch::Matchers::WithinAbs(1.0, 1E-6));
        REQUIRE_THAT(r().sd() + 1.0, Catch::Matchers::WithinAbs(1.0, 1E-6));
    }
}

} // namespace

TEMPLATE_LIST_TEST_CASE("Eigen", "", types2test) {
    using scalar_buffer = buffer::Eigen<TestType, 0>;
    using vector_buffer = buffer::Eigen<TestType, 1>;
    using matrix_buffer = buffer::Eigen<TestType, 2>;
    using tensor_buffer = buffer::Eigen<TestType, 3>;

    typename scalar_buffer::data_type eigen_scalar;
    eigen_scalar() = 10.0;

    typename vector_buffer::data_type eigen_vector(2);
    eigen_vector(0) = 10.0;
    eigen_vector(1) = 20.0;

    typename matrix_buffer::data_type eigen_matrix(2, 3);
    eigen_matrix(0, 0) = 10.0;
    eigen_matrix(0, 1) = 20.0;
    eigen_matrix(0, 2) = 30.0;
    eigen_matrix(1, 0) = 40.0;
    eigen_matrix(1, 1) = 50.0;
    eigen_matrix(1, 2) = 60.0;

    typename tensor_buffer::data_type eigen_tensor(1, 2, 3);
    eigen_tensor(0, 0, 0) = 10.0;
    eigen_tensor(0, 0, 1) = 20.0;
    eigen_tensor(0, 0, 2) = 30.0;
    eigen_tensor(0, 1, 0) = 40.0;
    eigen_tensor(0, 1, 1) = 50.0;
    eigen_tensor(0, 1, 2) = 60.0;

    auto scalar_layout = scalar_physical();
    auto vector_layout = vector_physical(2);
    auto matrix_layout = matrix_physical(2, 3);
    auto tensor_layout = tensor_physical(1, 2, 3);

    scalar_buffer scalar(eigen_scalar, scalar_layout);
    vector_buffer vector(eigen_vector, vector_layout);
    matrix_buffer matrix(eigen_matrix, matrix_layout);
    tensor_buffer tensor(eigen_tensor, tensor_layout);

    SECTION("ctors, assignment") {
        SECTION("value ctor") {
            compare_eigen<TestType>(scalar.value(), eigen_scalar);
            REQUIRE(scalar.layout().are_equal(scalar_layout));

            compare_eigen<TestType>(vector.value(), eigen_vector);
            REQUIRE(vector.layout().are_equal(vector_layout));

            compare_eigen<TestType>(matrix.value(), eigen_matrix);
            REQUIRE(matrix.layout().are_equal(matrix_layout));
        }

        test_copy_move_ctor_and_assignment(scalar, vector, matrix);
    }

    SECTION("value()") {
        compare_eigen<TestType>(scalar.value(), eigen_scalar);
        compare_eigen<TestType>(vector.value(), eigen_vector);
        compare_eigen<TestType>(matrix.value(), eigen_matrix);
    }

    SECTION("value() const") {
        const auto& cscalar = scalar;
        const auto& cvector = vector;
        const auto& cmatrix = matrix;
        compare_eigen<TestType>(cscalar.value(), eigen_scalar);
        compare_eigen<TestType>(cvector.value(), eigen_vector);
        compare_eigen<TestType>(cmatrix.value(), eigen_matrix);
    }

    SECTION("operator==") {
        // We assume the eigen tensor and the layout objects work. So when
        // comparing Eigen objects we have four states: same everything,
        // different everything, same tensor different layout, and different
        // tensor same layout.

        typename scalar_buffer::data_type eigen_scalar2;
        eigen_scalar2() = 10.0;

        // Everything the same
        REQUIRE(scalar == scalar_buffer(eigen_scalar2, scalar_layout));

        SECTION("Different scalar") {
            eigen_scalar2() = 2.0;
            scalar_buffer scalar2(eigen_scalar2, scalar_layout);
            REQUIRE_FALSE(scalar == scalar2);
        }

        SECTION("Different layout") {
            scalar_buffer scalar2(eigen_scalar, vector_layout);
            REQUIRE_FALSE(scalar == scalar2);
        }

        SECTION("Different tensor and layout") {
            eigen_scalar2() = 2.0;
            scalar_buffer scalar2(eigen_scalar2, vector_layout);
            REQUIRE_FALSE(scalar == scalar2);
        }
    }

    SECTION("operator!=") {
        // This just negates operator== so spot-checking is okay

        typename scalar_buffer::data_type eigen_scalar2;
        eigen_scalar2() = 10.0;

        // Everything the same
        scalar_buffer scalar2(eigen_scalar2, scalar_layout);
        REQUIRE_FALSE(scalar != scalar2);

        eigen_scalar2() = 2.0;
        scalar_buffer scalar3(eigen_scalar2, scalar_layout);
        REQUIRE(scalar3 != scalar);
    }

    SECTION("virtual method overrides") {
        using const_reference =
          typename scalar_buffer::const_buffer_base_reference;
        const_reference pscalar = scalar;
        const_reference pvector = vector;
        const_reference pmatrix = matrix;

        SECTION("clone") {
            REQUIRE(pscalar.clone()->are_equal(pscalar));
            REQUIRE(pvector.clone()->are_equal(pvector));
            REQUIRE(pmatrix.clone()->are_equal(pmatrix));
        }

        SECTION("are_equal") {
            scalar_buffer scalar2(eigen_scalar, scalar_layout);
            REQUIRE(pscalar.are_equal(scalar2));
            REQUIRE_FALSE(pmatrix.are_equal(scalar2));
        }
    }

    SECTION("addition_assignment_") {
        SECTION("scalar") {
            scalar_buffer scalar2(eigen_scalar, scalar_layout);
            scalar2.value()() = 42.0;

            auto s        = scalar("");
            auto pscalar2 = &(scalar2.addition_assignment("", s, s));

            scalar_buffer scalar_corr(eigen_scalar, scalar_layout);
            scalar_corr.value()() = 20.0;
            REQUIRE(pscalar2 == &scalar2);
            REQUIRE(scalar2 == scalar_corr);
        }

        SECTION("vector") {
            auto vector2 = testing::eigen_vector<TestType>();

            auto vi       = vector("i");
            auto pvector2 = &(vector2.addition_assignment("i", vi, vi));

            vector_buffer vector_corr(eigen_vector, vector_layout);
            vector_corr.value()(0) = 20.0;
            vector_corr.value()(1) = 40.0;

            REQUIRE(pvector2 == &vector2);
            REQUIRE(vector2 == vector_corr);
        }

        SECTION("matrix : no permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();

            auto mij      = matrix("i,j");
            auto pmatrix2 = &(matrix2.addition_assignment("i,j", mij, mij));

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 20.0;
            matrix_corr.value()(0, 1) = 40.0;
            matrix_corr.value()(0, 2) = 60.0;
            matrix_corr.value()(1, 0) = 80.0;
            matrix_corr.value()(1, 1) = 100.0;
            matrix_corr.value()(1, 2) = 120.0;

            REQUIRE(pmatrix2 == &matrix2);
            REQUIRE(matrix2 == matrix_corr);
        }

        SECTION("matrix: permutations") {
            auto matrix2 = testing::eigen_matrix<TestType>();
            auto l       = testing::matrix_physical(3, 2);
            std::array<int, 2> p10{1, 0};
            auto eigen_matrix_t = eigen_matrix.shuffle(p10);
            matrix_buffer matrix1(eigen_matrix_t, l);

            auto mij = matrix("i,j");
            auto mji = matrix1("j,i");

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 20.0;
            matrix_corr.value()(0, 1) = 40.0;
            matrix_corr.value()(0, 2) = 60.0;
            matrix_corr.value()(1, 0) = 80.0;
            matrix_corr.value()(1, 1) = 100.0;
            matrix_corr.value()(1, 2) = 120.0;

            SECTION("permute this") {
                matrix2.addition_assignment("j,i", mij, mij);

                matrix_buffer corr(eigen_matrix_t, l);
                corr.value()(0, 0) = 20.0;
                corr.value()(0, 1) = 80.0;
                corr.value()(1, 0) = 40.0;
                corr.value()(1, 1) = 100.0;
                corr.value()(2, 0) = 60.0;
                corr.value()(2, 1) = 120.0;

                REQUIRE(matrix2 == corr);
            }

            SECTION("permute LHS") {
                matrix2.addition_assignment("i,j", mji, mij);
                REQUIRE(matrix2 == matrix_corr);
            }

            SECTION("permute RHS") {
                matrix2.addition_assignment("i,j", mij, mji);
                REQUIRE(matrix2 == matrix_corr);
            }
        }

        SECTION("tensor (must permute all)") {
            auto tensor2 = testing::eigen_tensor3<TestType>();

            std::array<int, 3> p102{1, 0, 2};
            auto l102 = testing::tensor_physical(2, 1, 3);
            tensor_buffer tensor102(eigen_tensor.shuffle(p102), l102);

            auto tijk = tensor("i,j,k");
            auto tjik = tensor102("j,i,k");

            tensor2.addition_assignment("k,j,i", tijk, tjik);

            std::array<int, 3> p210{2, 1, 0};
            auto l210 = testing::tensor_physical(3, 2, 1);
            tensor_buffer corr(eigen_tensor.shuffle(p210), l210);
            corr.value()(0, 0, 0) = 20.0;
            corr.value()(0, 1, 0) = 80.0;
            corr.value()(1, 0, 0) = 40.0;
            corr.value()(1, 1, 0) = 100.0;
            corr.value()(2, 0, 0) = 60.0;
            corr.value()(2, 1, 0) = 120.0;
            REQUIRE(tensor2 == corr);
        }
    }

    SECTION("subtraction_assignment_") {
        SECTION("scalar") {
            scalar_buffer scalar2(eigen_scalar, scalar_layout);
            scalar2.value()() = 42.0;

            auto s        = scalar("");
            auto pscalar2 = &(scalar2.subtraction_assignment("", s, s));

            scalar_buffer scalar_corr(eigen_scalar, scalar_layout);
            scalar_corr.value()() = 0.0;
            REQUIRE(pscalar2 == &scalar2);
            REQUIRE(scalar2 == scalar_corr);
        }

        SECTION("vector") {
            auto vector2 = testing::eigen_vector<TestType>();

            auto vi       = vector("i");
            auto pvector2 = &(vector2.subtraction_assignment("i", vi, vi));

            vector_buffer vector_corr(eigen_vector, vector_layout);
            vector_corr.value()(0) = 0.0;
            vector_corr.value()(1) = 0.0;

            REQUIRE(pvector2 == &vector2);
            REQUIRE(vector2 == vector_corr);
        }

        SECTION("matrix : no permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();

            auto mij      = matrix("i,j");
            auto pmatrix2 = &(matrix2.subtraction_assignment("i,j", mij, mij));

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 0.0;
            matrix_corr.value()(0, 1) = 0.0;
            matrix_corr.value()(0, 2) = 0.0;
            matrix_corr.value()(1, 0) = 0.0;
            matrix_corr.value()(1, 1) = 0.0;
            matrix_corr.value()(1, 2) = 0.0;

            REQUIRE(pmatrix2 == &matrix2);
            REQUIRE(matrix2 == matrix_corr);
        }

        SECTION("matrix: permutations") {
            auto matrix2 = testing::eigen_matrix<TestType>();
            auto l       = testing::matrix_physical(3, 2);
            std::array<int, 2> p10{1, 0};
            auto eigen_matrix_t = eigen_matrix.shuffle(p10);
            matrix_buffer matrix1(eigen_matrix_t, l);

            auto mij = matrix("i,j");
            auto mji = matrix1("j,i");

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 0.0;
            matrix_corr.value()(0, 1) = 0.0;
            matrix_corr.value()(0, 2) = 0.0;
            matrix_corr.value()(1, 0) = 0.0;
            matrix_corr.value()(1, 1) = 0.0;
            matrix_corr.value()(1, 2) = 0.0;

            SECTION("permute this") {
                matrix2.subtraction_assignment("j,i", mij, mij);

                matrix_buffer corr(eigen_matrix_t, l);
                corr.value()(0, 0) = 0.0;
                corr.value()(0, 1) = 0.0;
                corr.value()(1, 0) = 0.0;
                corr.value()(1, 1) = 0.0;
                corr.value()(2, 0) = 0.0;
                corr.value()(2, 1) = 0.0;

                REQUIRE(matrix2 == corr);
            }

            SECTION("permute LHS") {
                matrix2.subtraction_assignment("i,j", mji, mij);
                REQUIRE(matrix2 == matrix_corr);
            }

            SECTION("permute RHS") {
                matrix2.subtraction_assignment("i,j", mij, mji);
                REQUIRE(matrix2 == matrix_corr);
            }
        }

        SECTION("tensor (must permute all)") {
            auto tensor2 = testing::eigen_tensor3<TestType>();

            std::array<int, 3> p102{1, 0, 2};
            auto l102 = testing::tensor_physical(2, 1, 3);
            tensor_buffer tensor102(eigen_tensor.shuffle(p102), l102);

            auto tijk = tensor("i,j,k");
            auto tjik = tensor102("j,i,k");

            tensor2.subtraction_assignment("k,j,i", tijk, tjik);

            std::array<int, 3> p210{2, 1, 0};
            auto l210 = testing::tensor_physical(3, 2, 1);
            tensor_buffer corr(eigen_tensor.shuffle(p210), l210);
            corr.value()(0, 0, 0) = 0.0;
            corr.value()(0, 1, 0) = 0.0;
            corr.value()(1, 0, 0) = 0.0;
            corr.value()(1, 1, 0) = 0.0;
            corr.value()(2, 0, 0) = 0.0;
            corr.value()(2, 1, 0) = 0.0;
            REQUIRE(tensor2 == corr);
        }
    }

    SECTION("multiplication_assignment_") {
        // Multiplication just dispatches to hadamard_ or contraction_
        // Here we test the error-handling

        // Must be either a pure hadamard or a pure contraction
        auto matrix2 = testing::eigen_matrix<TestType>();
        auto mij     = matrix("i,j");

        REQUIRE_THROWS_AS(matrix2.subtraction_assignment("i", mij, mij),
                          std::runtime_error);
    }

    SECTION("permute_assignment_") {
        SECTION("scalar") {
            auto scalar2      = testing::eigen_scalar<TestType>();
            scalar2.value()() = 42.0;

            auto s        = scalar("");
            auto pscalar2 = &(scalar2.permute_assignment("", s));
            REQUIRE(pscalar2 == &scalar2);
            REQUIRE(scalar2 == scalar);
        }

        SECTION("vector") {
            auto vector2 = testing::eigen_vector<TestType>();

            auto vi       = vector("i");
            auto pvector2 = &(vector2.permute_assignment("i", vi));

            REQUIRE(pvector2 == &vector2);
            REQUIRE(vector2 == vector);
        }

        SECTION("matrix : no permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();

            auto mij      = matrix("i,j");
            auto pmatrix2 = &(matrix2.permute_assignment("i,j", mij));

            REQUIRE(pmatrix2 == &matrix2);
            REQUIRE(matrix2 == matrix);
        }

        SECTION("matrix: permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();
            auto p       = &(matrix2.permute_assignment("j,i", matrix("i,j")));

            auto corr          = testing::eigen_matrix<TestType>(3, 2);
            corr.value()(0, 0) = 10.0;
            corr.value()(1, 0) = 20.0;
            corr.value()(2, 0) = 30.0;
            corr.value()(0, 1) = 40.0;
            corr.value()(1, 1) = 50.0;
            corr.value()(2, 1) = 60.0;
            REQUIRE(p == &matrix2);
            compare_eigen<TestType>(corr.value(), matrix2.value());
        }
    }

    SECTION("scalar_multiplication_") {
        SECTION("scalar") {
            auto scalar2      = testing::eigen_scalar<TestType>();
            scalar2.value()() = 42.0;

            auto s        = scalar("");
            auto pscalar2 = &(scalar2.scalar_multiplication("", 2.0, s));

            auto corr      = testing::eigen_scalar<TestType>();
            corr.value()() = 20.0;
            REQUIRE(pscalar2 == &scalar2);
            REQUIRE(scalar2 == corr);
        }

        SECTION("vector") {
            auto vector2 = testing::eigen_vector<TestType>();

            auto vi       = vector("i");
            auto pvector2 = &(vector2.scalar_multiplication("i", 2.0, vi));

            auto corr       = testing::eigen_vector<TestType>(2);
            corr.value()(0) = 20.0;
            corr.value()(1) = 40.0;

            REQUIRE(pvector2 == &vector2);
            REQUIRE(vector2 == corr);
        }

        SECTION("matrix : no permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();

            auto mij = matrix("i,j");
            auto p   = &(matrix2.scalar_multiplication("i,j", 2.0, mij));

            auto corr          = testing::eigen_matrix<TestType>(2, 3);
            corr.value()(0, 0) = 20.0;
            corr.value()(0, 1) = 40.0;
            corr.value()(0, 2) = 60.0;
            corr.value()(1, 0) = 80.0;
            corr.value()(1, 1) = 100.0;
            corr.value()(1, 2) = 120.0;

            REQUIRE(p == &matrix2);
            REQUIRE(matrix2 == corr);
        }

        SECTION("matrix: permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();
            auto mij     = matrix("i,j");
            auto p       = &(matrix2.scalar_multiplication("j,i", 2.0, mij));

            auto corr          = testing::eigen_matrix<TestType>(3, 2);
            corr.value()(0, 0) = 20.0;
            corr.value()(1, 0) = 40.0;
            corr.value()(2, 0) = 60.0;
            corr.value()(0, 1) = 80.0;
            corr.value()(1, 1) = 100.0;
            corr.value()(2, 1) = 120.0;
            REQUIRE(p == &matrix2);
            compare_eigen<TestType>(corr.value(), matrix2.value());
        }
    }

    SECTION("hadamard_") {
        SECTION("scalar") {
            scalar_buffer scalar2(eigen_scalar, scalar_layout);
            scalar2.value()() = 42.0;

            auto s        = scalar("");
            auto pscalar2 = &(scalar2.multiplication_assignment("", s, s));

            scalar_buffer scalar_corr(eigen_scalar, scalar_layout);
            scalar_corr.value()() = 100.0;
            REQUIRE(pscalar2 == &scalar2);
            REQUIRE(scalar2 == scalar_corr);
        }

        SECTION("vector") {
            auto vector2 = testing::eigen_vector<TestType>();

            auto vi       = vector("i");
            auto pvector2 = &(vector2.multiplication_assignment("i", vi, vi));

            vector_buffer vector_corr(eigen_vector, vector_layout);
            vector_corr.value()(0) = 100.0;
            vector_corr.value()(1) = 400.0;

            REQUIRE(pvector2 == &vector2);
            REQUIRE(vector2 == vector_corr);
        }

        SECTION("matrix : no permutation") {
            auto matrix2 = testing::eigen_matrix<TestType>();

            auto mij = matrix("i,j");
            auto pmatrix2 =
              &(matrix2.multiplication_assignment("i,j", mij, mij));

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 100.0;
            matrix_corr.value()(0, 1) = 400.0;
            matrix_corr.value()(0, 2) = 900.0;
            matrix_corr.value()(1, 0) = 1600.0;
            matrix_corr.value()(1, 1) = 2500.0;
            matrix_corr.value()(1, 2) = 3600.0;

            REQUIRE(pmatrix2 == &matrix2);
            REQUIRE(matrix2 == matrix_corr);
        }

        SECTION("matrix: permutations") {
            auto matrix2 = testing::eigen_matrix<TestType>();
            auto l       = testing::matrix_physical(3, 2);
            std::array<int, 2> p10{1, 0};
            auto eigen_matrix_t = eigen_matrix.shuffle(p10);
            matrix_buffer matrix1(eigen_matrix_t, l);

            auto mij = matrix("i,j");
            auto mji = matrix1("j,i");

            matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

            matrix_corr.value()(0, 0) = 100.0;
            matrix_corr.value()(0, 1) = 400.0;
            matrix_corr.value()(0, 2) = 900.0;
            matrix_corr.value()(1, 0) = 1600.0;
            matrix_corr.value()(1, 1) = 2500.0;
            matrix_corr.value()(1, 2) = 3600.0;

            SECTION("permute this") {
                matrix2.multiplication_assignment("j,i", mij, mij);

                matrix_buffer corr(eigen_matrix_t, l);
                corr.value()(0, 0) = 100.0;
                corr.value()(0, 1) = 1600.0;
                corr.value()(1, 0) = 400.0;
                corr.value()(1, 1) = 2500.0;
                corr.value()(2, 0) = 900.0;
                corr.value()(2, 1) = 3600.0;

                REQUIRE(matrix2 == corr);
            }

            SECTION("permute LHS") {
                matrix2.multiplication_assignment("i,j", mji, mij);
                REQUIRE(matrix2 == matrix_corr);
            }

            SECTION("permute RHS") {
                matrix2.multiplication_assignment("i,j", mij, mji);
                REQUIRE(matrix2 == matrix_corr);
            }
        }

        SECTION("tensor (must permute all)") {
            auto tensor2 = testing::eigen_tensor3<TestType>();

            std::array<int, 3> p102{1, 0, 2};
            auto l102 = testing::tensor_physical(2, 1, 3);
            tensor_buffer tensor102(eigen_tensor.shuffle(p102), l102);

            auto tijk = tensor("i,j,k");
            auto tjik = tensor102("j,i,k");

            tensor2.multiplication_assignment("k,j,i", tijk, tjik);

            std::array<int, 3> p210{2, 1, 0};
            auto l210 = testing::tensor_physical(3, 2, 1);
            tensor_buffer corr(eigen_tensor.shuffle(p210), l210);
            corr.value()(0, 0, 0) = 100.0;
            corr.value()(0, 1, 0) = 1600.0;
            corr.value()(1, 0, 0) = 400.0;
            corr.value()(1, 1, 0) = 2500.0;
            corr.value()(2, 0, 0) = 900.0;
            corr.value()(2, 1, 0) = 3600.0;
            REQUIRE(tensor2 == corr);
        }
    }

    SECTION("contraction_") {
        auto vi  = vector("i");
        auto mij = matrix("i,j");
        auto mik = matrix("i,k");
        auto mjk = matrix("j,k");

        SECTION("vector with vector") {
            auto p = &(scalar.multiplication_assignment("", vi, vi));

            auto scalar_corr      = testing::eigen_scalar<TestType>();
            scalar_corr.value()() = 500.0; // 10*10 + 20*20
            REQUIRE(p == &scalar);
            compare_eigen<TestType>(scalar_corr.value(), scalar.value());
        }

        SECTION("ij,ij->") {
            auto p = &(scalar.multiplication_assignment("", mij, mij));

            auto scalar_corr      = testing::eigen_scalar<TestType>();
            scalar_corr.value()() = 9100.0; // 1400 + 7700
            REQUIRE(p == &scalar);
            compare_eigen<TestType>(scalar_corr.value(), scalar.value());
        }

        SECTION("ki,kj->ij") {
            auto buffer2 = testing::eigen_matrix<TestType>();
            auto p = &(buffer2.multiplication_assignment("i,j", mik, mjk));

            auto matrix_corr          = testing::eigen_matrix<TestType>(2, 2);
            matrix_corr.value()(0, 0) = 1400.0; // 100 + 400 + 900
            matrix_corr.value()(0, 1) = 3200.0; // 400 + 1000 + 1800
            matrix_corr.value()(1, 0) = 3200.0; // 400 + 1000 + 1800
            matrix_corr.value()(1, 1) = 7700.0; // 1600 + 2500 + 3600

            REQUIRE(p == &buffer2);
            compare_eigen<TestType>(matrix_corr.value(), buffer2.value());
        }

        SECTION("ij,i->j") {
            auto buffer1 = testing::eigen_vector<TestType>();
            auto p       = &(buffer1.multiplication_assignment("j", mij,
            vi));

            auto vector_corr       = testing::eigen_vector<TestType>(3);
            vector_corr.value()(0) = 900.0;  // 10(10) + 20(40)
            vector_corr.value()(1) = 1200.0; // 10(20) + 20(50)
            vector_corr.value()(2) = 1500.0; // 10(30) + 20(60)
            REQUIRE(p == &buffer1);
            compare_eigen<TestType>(vector_corr.value(), buffer1.value());
        }
    }
}
