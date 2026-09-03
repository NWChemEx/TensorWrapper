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
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/buffer/buffer_view_base.hpp>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace buffer;
using testing::TestView;

TEST_CASE("BufferBaseCommon") {
    using MutableView = BufferViewBase<BufferBase>;
    using ConstView   = BufferViewBase<const BufferBase>;

    auto pscalar = testing::eigen_scalar<double>();
    auto& scalar = *pscalar;
    scalar.set_elem({}, 1.0);

    auto pvector = testing::eigen_vector<double>(2);
    auto& vector = *pvector;
    vector.set_elem({0}, 1.0);
    vector.set_elem({1}, 2.0);

    auto scalar_layout = testing::scalar_physical();
    auto vector_layout = testing::vector_physical(2);

    buffer::Contiguous defaulted;

    // Concrete objects (BufferViewBase itself is abstract); the rest of this
    // test only ever interacts with them through BufferViewBase references.
    TestView<BufferBase> defaulted_view_impl(defaulted);
    TestView<BufferBase> scalar_view_impl(scalar);
    TestView<BufferBase> vector_view_impl(vector);
    TestView<const BufferBase> defaulted_const_view_impl(defaulted);
    TestView<const BufferBase> scalar_const_view_impl(scalar);
    TestView<const BufferBase> vector_const_view_impl(vector);

    MutableView& defaulted_view     = defaulted_view_impl;
    MutableView& scalar_view        = scalar_view_impl;
    MutableView& vector_view        = vector_view_impl;
    ConstView& defaulted_const_view = defaulted_const_view_impl;
    ConstView& scalar_const_view    = scalar_const_view_impl;
    ConstView& vector_const_view    = vector_const_view_impl;

    SECTION("operator== (BufferBase with BufferBaseView)") {
        REQUIRE(defaulted_view == defaulted);
        REQUIRE(defaulted_const_view == defaulted);
        REQUIRE(defaulted == defaulted_view);
        REQUIRE(defaulted == defaulted_const_view);

        REQUIRE_FALSE(defaulted_view == scalar);
        REQUIRE_FALSE(defaulted_const_view == scalar);
        REQUIRE_FALSE(scalar == defaulted_view);
        REQUIRE_FALSE(scalar == defaulted_const_view);
    }

    SECTION("operator!= (BufferBasewith BufferBase") {
        REQUIRE(scalar_view != vector);
        REQUIRE(scalar_const_view != vector);
        REQUIRE(vector != scalar_view);
        REQUIRE(vector != scalar_const_view);

        REQUIRE(scalar_view != defaulted);
        REQUIRE(scalar_const_view != defaulted);
        REQUIRE(defaulted != scalar_view);
        REQUIRE(defaulted != scalar_const_view);

        REQUIRE_FALSE(scalar_view != scalar);
        REQUIRE_FALSE(scalar_const_view != scalar);
        REQUIRE_FALSE(scalar != scalar_view);
        REQUIRE_FALSE(scalar != scalar_const_view);
    }

    SECTION("BufferBase operator== with BufferBase") {
        REQUIRE(scalar == scalar);
        REQUIRE_FALSE(scalar == vector);
        REQUIRE_FALSE(defaulted == scalar);
    }

    SECTION("BufferViewBase operator== with BufferViewBase") {
        REQUIRE(scalar_view == scalar_view);
        REQUIRE(scalar_const_view == scalar_const_view);
        REQUIRE(scalar_view == scalar_const_view);
        REQUIRE(scalar_const_view == scalar_view);
        REQUIRE_FALSE(scalar_view == vector_view);
        REQUIRE_FALSE(defaulted_view == scalar_view);
        REQUIRE_FALSE(scalar_const_view == vector_view);
        REQUIRE_FALSE(vector_view == scalar_const_view);
        REQUIRE_FALSE(defaulted_view == vector_view);
        REQUIRE_FALSE(vector_view == defaulted_view);
    }

    SECTION("approximately_equal") {
        REQUIRE(scalar.approximately_equal(scalar, 1e-10));
        REQUIRE(scalar_view.approximately_equal(scalar_view, 1e-10));
        REQUIRE(scalar_view.approximately_equal(scalar, 1e-10));
        REQUIRE(scalar.approximately_equal(scalar_view, 1e-10));

        REQUIRE_FALSE(scalar_view.approximately_equal(vector, 1e-10));
        REQUIRE_FALSE(vector.approximately_equal(scalar_view, 1e-10));
    }

    SECTION("Null view equals buffer with no layout") {
        TestView<const BufferBase> null_view_impl;
        ConstView& null_view = null_view_impl;
        REQUIRE(null_view == defaulted);
        REQUIRE_FALSE(null_view == scalar);
    }

    SECTION("get_element/set_element (index_vector overload)") {
        REQUIRE(scalar.get_element({}) == scalar.get_elem({}));
        REQUIRE(vector.get_element({0}) == vector.get_elem({0}));
        REQUIRE(vector.get_element({1}) == vector.get_elem({1}));

        scalar.set_element({}, 2.0);
        REQUIRE(scalar.get_elem({}) == 2.0);

        vector.set_element({0}, 9.0);
        REQUIRE(vector.get_elem({0}) == 9.0);
    }

    SECTION("get_element/set_element (variadic offset overload)") {
        REQUIRE(scalar.get_element() == scalar.get_elem({}));
        REQUIRE(vector.get_element(0) == vector.get_elem({0}));
        REQUIRE(vector.get_element(1) == vector.get_elem({1}));

        scalar.set_element(3.0);
        REQUIRE(scalar.get_elem({}) == 3.0);

        vector.set_element(0, 8.0);
        REQUIRE(vector.get_elem({0}) == 8.0);

        // Wrong number of offsets for the rank still throws, proving the
        // variadic overload delegates to (and does not bypass) the existing
        // rank/bounds check.
        REQUIRE_THROWS_AS(vector.get_element(0, 1), std::out_of_range);
        REQUIRE_THROWS_AS(vector.set_element(0, 1, 1.0), std::out_of_range);
    }

    SECTION("get_element/set_element on an unsupported view throws") {
        REQUIRE_THROWS_AS(scalar_view.get_element({}), std::runtime_error);
        REQUIRE_THROWS_AS(scalar_view.set_element({}, 1.0), std::runtime_error);
        REQUIRE_THROWS_AS(scalar_const_view.get_element({}),
                          std::runtime_error);
    }
}
