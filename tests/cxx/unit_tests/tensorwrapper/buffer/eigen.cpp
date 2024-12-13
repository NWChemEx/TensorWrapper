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

#include "../helpers.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace testing;

namespace {

template<typename FloatType, typename LHSType, typename RHSType>
bool compare_eigen(const LHSType& lhs, const RHSType& rhs) {
    using r_type = Eigen::Tensor<FloatType, 0, Eigen::RowMajor>;
    auto d       = lhs - rhs;
    r_type r     = d.sum();

    return (r() == 0.0);
}

} // namespace

TEMPLATE_TEST_CASE("Eigen", "", float, double) {
    if constexpr(have_eigen()) {
        using scalar_buffer = buffer::Eigen<TestType, 0>;
        using vector_buffer = buffer::Eigen<TestType, 1>;
        using matrix_buffer = buffer::Eigen<TestType, 2>;

        typename scalar_buffer::data_type eigen_scalar;
        eigen_scalar() = 1.0;

        typename vector_buffer::data_type eigen_vector(2);
        eigen_vector(0) = 1.0;
        eigen_vector(1) = 2.0;

        typename matrix_buffer::data_type eigen_matrix(2, 3);
        eigen_matrix(0, 0) = 1.0;
        eigen_matrix(0, 1) = 2.0;
        eigen_matrix(0, 2) = 3.0;
        eigen_matrix(1, 0) = 4.0;
        eigen_matrix(1, 1) = 5.0;
        eigen_matrix(1, 2) = 6.0;

        symmetry::Group g;
        sparsity::Pattern p;
        layout::Physical scalar_layout(shape::Smooth{}, g, p);
        layout::Physical vector_layout(shape::Smooth{2}, g, p);
        layout::Physical matrix_layout(shape::Smooth{2, 3}, g, p);

        scalar_buffer scalar(eigen_scalar, scalar_layout);
        vector_buffer vector(eigen_vector, vector_layout);
        matrix_buffer matrix(eigen_matrix, matrix_layout);

        SECTION("ctors, assignment") {
            SECTION("value ctor") {
                REQUIRE(compare_eigen<TestType>(scalar.value(), eigen_scalar));
                REQUIRE(scalar.layout().are_equal(scalar_layout));

                REQUIRE(compare_eigen<TestType>(vector.value(), eigen_vector));
                REQUIRE(vector.layout().are_equal(vector_layout));

                REQUIRE(compare_eigen<TestType>(matrix.value(), eigen_matrix));
                REQUIRE(matrix.layout().are_equal(matrix_layout));
            }

            test_copy_move_ctor_and_assignment(scalar, vector, matrix);
        }

        SECTION("value()") {
            REQUIRE(compare_eigen<TestType>(scalar.value(), eigen_scalar));
            REQUIRE(compare_eigen<TestType>(vector.value(), eigen_vector));
            REQUIRE(compare_eigen<TestType>(matrix.value(), eigen_matrix));
        }

        SECTION("value() const") {
            const auto& cscalar = scalar;
            const auto& cvector = vector;
            const auto& cmatrix = matrix;
            REQUIRE(compare_eigen<TestType>(cscalar.value(), eigen_scalar));
            REQUIRE(compare_eigen<TestType>(cvector.value(), eigen_vector));
            REQUIRE(compare_eigen<TestType>(cmatrix.value(), eigen_matrix));
        }

        SECTION("operator==") {
            // We assume the eigen tensor and the layout objects work. So when
            // comparing Eigen objects we have four states: same everything,
            // different everything, same tensor different layout, and different
            // tensor same layout.

            typename scalar_buffer::data_type eigen_scalar2;
            eigen_scalar2() = 1.0;

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
            eigen_scalar2() = 1.0;

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

            SECTION("addition_assignment") {
                SECTION("scalar") {
                    scalar_buffer scalar2(eigen_scalar, scalar_layout);
                    scalar2.value()() = 42.0;

                    auto s        = scalar("");
                    auto pscalar2 = &(scalar2.addition_assignment("", s));

                    scalar_buffer scalar_corr(eigen_scalar, scalar_layout);
                    scalar_corr.value()() = 43.0;
                    REQUIRE(pscalar2 == &scalar2);
                    REQUIRE(scalar2 == scalar_corr);
                }

                SECTION("vector") {
                    vector_buffer vector2(eigen_vector, vector_layout);

                    auto vi       = vector("i");
                    auto pvector2 = &(vector2.addition_assignment("i", vi));

                    vector_buffer vector_corr(eigen_vector, vector_layout);
                    vector_corr.value()(0) = 2.0;
                    vector_corr.value()(1) = 4.0;

                    REQUIRE(pvector2 == &vector2);
                    REQUIRE(vector2 == vector_corr);
                }

                SECTION("matrix : no permutation") {
                    matrix_buffer matrix2(eigen_matrix, matrix_layout);

                    auto mij      = matrix("i,j");
                    auto pmatrix2 = &(matrix2.addition_assignment("i,j", mij));

                    matrix_buffer matrix_corr(eigen_matrix, matrix_layout);

                    matrix_corr.value()(0, 0) = 2.0;
                    matrix_corr.value()(0, 1) = 4.0;
                    matrix_corr.value()(0, 2) = 6.0;
                    matrix_corr.value()(1, 0) = 8.0;
                    matrix_corr.value()(1, 1) = 10.0;
                    matrix_corr.value()(1, 2) = 12.0;

                    REQUIRE(pmatrix2 == &matrix2);
                    REQUIRE(matrix2 == matrix_corr);
                }

                SECTION("matrix : permutation") {
                    layout::Physical l(shape::Smooth{3, 2}, g, p);
                    std::array<int, 2> p10{1, 0};
                    auto eigen_matrix_t = eigen_matrix.shuffle(p10);
                    matrix_buffer matrix2(eigen_matrix_t, l);
                    auto mij      = matrix("i,j");
                    auto pmatrix2 = &(matrix2.addition_assignment("j,i", mij));

                    matrix_buffer corr(eigen_matrix_t, l);
                    corr.value()(0, 0) = 2.0;
                    corr.value()(0, 1) = 8.0;
                    corr.value()(1, 0) = 4.0;
                    corr.value()(1, 1) = 10.0;
                    corr.value()(2, 0) = 6.0;
                    corr.value()(2, 1) = 12.0;

                    REQUIRE(pmatrix2 == &matrix2);
                    REQUIRE(matrix2 == corr);
                }

                // Can't cast
                REQUIRE_THROWS_AS(vector.addition_assignment("", scalar("")),
                                  std::runtime_error);

                // Labels must match
                REQUIRE_THROWS_AS(vector.addition_assignment("j", vector("i")),
                                  std::runtime_error);
            }

            SECTION("permute_assignment") {
                SECTION("scalar") {
                    scalar_buffer scalar2;
                    auto s        = scalar("");
                    auto pscalar2 = &(scalar2.permute_assignment("", s));
                    REQUIRE(pscalar2 == &scalar2);
                    REQUIRE(scalar2 == scalar);
                }

                SECTION("vector") {
                    vector_buffer vector2;
                    auto vi       = vector("i");
                    auto pvector2 = &(vector2.permute_assignment("i", vi));
                    REQUIRE(pvector2 == &vector2);
                    REQUIRE(vector2 == vector);
                }

                SECTION("matrix : no permutation") {
                    matrix_buffer matrix2;
                    auto mij      = matrix("i,j");
                    auto pmatrix2 = &(matrix2.permute_assignment("i,j", mij));
                    REQUIRE(pmatrix2 == &matrix2);
                    REQUIRE(matrix2 == matrix);
                }
                SECTION("matrix : permutation") {
                    matrix_buffer matrix2;
                    auto mij      = matrix("i,j");
                    auto pmatrix2 = &(matrix2.permute_assignment("j,i", mij));

                    layout::Physical l(shape::Smooth{3, 2}, g, p);
                    std::array<int, 2> p10{1, 0};
                    auto eigen_matrix_t = eigen_matrix.shuffle(p10);
                    matrix_buffer corr(eigen_matrix_t, l);
                    REQUIRE(pmatrix2 == &matrix2);
                    REQUIRE(matrix2 == corr);
                }
            }
        }
    }
}
