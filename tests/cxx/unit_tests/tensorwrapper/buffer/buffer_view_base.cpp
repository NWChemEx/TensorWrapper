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
#include <tensorwrapper/buffer/buffer_view_base.hpp>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace buffer;
using testing::TestView;

TEST_CASE("BufferViewBase") {
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

    SECTION("Default construction") {
        TestView<const BufferBase> defaulted_const_view_impl;
        TestView<BufferBase> defaulted_view_impl;
        ConstView& defaulted_const_view = defaulted_const_view_impl;
        MutableView& defaulted_view     = defaulted_view_impl;
        REQUIRE_FALSE(defaulted_const_view.has_layout());
        REQUIRE_FALSE(defaulted_view.has_layout());
        REQUIRE_THROWS_AS(defaulted_const_view.layout(), std::runtime_error);
        REQUIRE_THROWS_AS(defaulted_view.layout(), std::runtime_error);
        REQUIRE(defaulted_const_view.rank() == 0);
        REQUIRE(defaulted_view.rank() == 0);
    }

    SECTION("Construct from buffer") {
        TestView<const BufferBase> scalar_const_view_impl(scalar);
        TestView<BufferBase> scalar_view_impl(scalar);
        ConstView& scalar_const_view = scalar_const_view_impl;
        MutableView& scalar_view     = scalar_view_impl;
        REQUIRE(scalar_const_view.has_layout());
        REQUIRE(scalar_view.has_layout());
        REQUIRE(scalar_const_view.layout().are_equal(scalar_layout));
        REQUIRE(scalar_const_view.rank() == 0);
        REQUIRE(scalar_view.layout().are_equal(scalar_layout));
        REQUIRE(scalar_view.rank() == 0);

        TestView<const BufferBase> vector_const_view_impl(vector);
        TestView<BufferBase> vector_view_impl(vector);
        ConstView& vector_const_view = vector_const_view_impl;
        MutableView& vector_view     = vector_view_impl;
        REQUIRE(vector_const_view.has_layout());
        REQUIRE(vector_view.has_layout());
        REQUIRE(vector_const_view.layout().are_equal(vector_layout));
        REQUIRE(vector_const_view.rank() == 1);
        REQUIRE(vector_view.layout().are_equal(vector_layout));
        REQUIRE(vector_view.rank() == 1);
    }

    SECTION("Copy construction") {
        TestView<const BufferBase> const_view(scalar);
        TestView<const BufferBase> copy_const(const_view);
        REQUIRE(copy_const.has_layout());
        REQUIRE(copy_const.layout().are_equal(scalar_layout));
        REQUIRE(copy_const.rank() == 0);

        TestView<BufferBase> mutable_view(scalar);
        TestView<BufferBase> copy_mutable(mutable_view);
        REQUIRE(copy_mutable.has_layout());
        REQUIRE(copy_mutable.layout().are_equal(scalar_layout));
        REQUIRE(copy_mutable.rank() == 0);
    }

    SECTION("Move construction") {
        TestView<const BufferBase> const_view(scalar);
        TestView<const BufferBase> moved_const(std::move(const_view));
        REQUIRE(moved_const.has_layout());
        REQUIRE(moved_const.layout().are_equal(scalar_layout));
        REQUIRE(moved_const.rank() == 0);

        TestView<BufferBase> mutable_view(scalar);
        TestView<BufferBase> moved(std::move(mutable_view));
        REQUIRE(moved.has_layout());
        REQUIRE(moved.layout().are_equal(scalar_layout));
        REQUIRE(moved.rank() == 0);
    }

    SECTION("Copy assignment") {
        TestView<const BufferBase> const_view(scalar);
        TestView<const BufferBase> other_const_impl;
        ConstView& other_const = other_const_impl;
        auto pother_const      = &(other_const = const_view);
        REQUIRE(pother_const == &other_const);
        REQUIRE(other_const.has_layout());
        REQUIRE(other_const.layout().are_equal(scalar_layout));
        REQUIRE(other_const.rank() == 0);

        TestView<BufferBase> mutable_view(scalar);
        TestView<BufferBase> other_impl;
        MutableView& other = other_impl;
        other              = mutable_view;
        REQUIRE(other.has_layout());
        REQUIRE(other.layout().are_equal(scalar_layout));
        REQUIRE(other.rank() == 0);
    }

    SECTION("Move assignment") {
        TestView<const BufferBase> const_view(scalar);
        TestView<const BufferBase> other_const_impl;
        ConstView& other_const = other_const_impl;
        auto pother_const      = &(other_const = std::move(const_view));
        REQUIRE(pother_const == &other_const);
        REQUIRE(other_const.has_layout());
        REQUIRE(other_const.layout().are_equal(scalar_layout));
        REQUIRE(other_const.rank() == 0);

        TestView<BufferBase> mutable_view(scalar);
        TestView<BufferBase> other_mutable_impl;
        MutableView& other_mutable = other_mutable_impl;
        other_mutable              = std::move(mutable_view);
        REQUIRE(other_mutable.has_layout());
        REQUIRE(other_mutable.layout().are_equal(scalar_layout));
        REQUIRE(other_mutable.rank() == 0);
    }
}
