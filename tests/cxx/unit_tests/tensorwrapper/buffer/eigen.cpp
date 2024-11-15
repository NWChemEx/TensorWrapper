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
        }
    }
}
