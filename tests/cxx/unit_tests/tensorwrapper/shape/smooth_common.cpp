/*
 * Copyright 2026 NWChemEx-Project
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
#include <stdexcept>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/shape/smooth_common.hpp>

namespace {

/* All of these functions expect a vector of shapes or views of shapes such
 * that:
 *
 * - Element 0 is the empty shape.
 * - Element 1 is a scalar shape.
 * - Element 2 is a vector shape.
 * - Element 3 is a matrix shape.
 * - Element 4 is a rank 3 tensor shape.
 */

template<typename ShapeType>
void test_extent(std::vector<ShapeType> shapes) {
    SECTION("Empty Shape") {
        REQUIRE_THROWS_AS(shapes[0].extent(0), std::out_of_range);
    }

    SECTION("Scalar Shape") {
        REQUIRE_THROWS_AS(shapes[1].extent(0), std::out_of_range);
    }

    SECTION("Vector Shape") {
        REQUIRE(shapes[2].extent(0) == 3);
        REQUIRE_THROWS_AS(shapes[2].extent(1), std::out_of_range);
    }

    SECTION("Matrix Shape") {
        REQUIRE(shapes[3].extent(0) == 2);
        REQUIRE(shapes[3].extent(1) == 4);
        REQUIRE_THROWS_AS(shapes[3].extent(2), std::out_of_range);
    }

    SECTION("Tensor Shape") {
        REQUIRE(shapes[4].extent(0) == 4);
        REQUIRE(shapes[4].extent(1) == 5);
        REQUIRE(shapes[4].extent(2) == 6);
        REQUIRE_THROWS_AS(shapes[4].extent(3), std::out_of_range);
    }
}

template<typename ShapeType>
void test_slice_il(std::vector<ShapeType> shapes) {
    // N.b. we just spot check here, full checking happens in the range test

    using smooth_type = tensorwrapper::shape::Smooth;

    auto vslice = shapes[2].slice({0}, {2});
    REQUIRE(vslice == smooth_type{2});

    auto mslice = shapes[3].slice({1, 1}, {3, 2});
    REQUIRE(mslice == smooth_type{2, 1});

    auto tslice = shapes[4].slice({0, 0, 0}, {4, 5, 6});
    REQUIRE(tslice == shapes[4]);
}

template<typename ShapeType>
void test_slice_container(std::vector<ShapeType> shapes) {
    // N.b. we just spot check here, full checking happens in the range test

    using smooth_type = tensorwrapper::shape::Smooth;
    using size_type   = typename smooth_type::size_type;
    using size_vector = std::vector<size_type>;

    size_vector i0{0}, i2{2};
    auto vslice = shapes[2].slice(i0, i2);
    REQUIRE(vslice == smooth_type{2});

    size_vector i11{1, 1}, i32{3, 2};
    auto mslice = shapes[3].slice(i11, i32);
    REQUIRE(mslice == smooth_type{2, 1});

    size_vector i000{0, 0, 0}, i456{4, 5, 6};
    auto tslice = shapes[4].slice(i000, i456);
    REQUIRE(tslice == shapes[4]);
}

template<typename ShapeType>
void test_slice_ranges(std::vector<ShapeType> shapes) {
    using smooth_type = tensorwrapper::shape::Smooth;
    using size_type   = typename smooth_type::size_type;
    using size_vector = std::vector<size_type>;

    size_vector empty;
    auto eb = empty.begin();
    auto ee = empty.end();

    size_vector i0{0}, i2{2};
    auto i0b = i0.begin();
    auto i0e = i0.end();
    auto i2b = i2.begin();
    auto i2e = i2.end();

    size_vector i11{1, 1}, i32{3, 2};
    auto i11b = i11.begin();
    auto i11e = i11.end();
    auto i32b = i32.begin();
    auto i32e = i32.end();

    size_vector i000{0, 0, 0}, i456{4, 5, 6};
    auto i000b = i000.begin();
    auto i000e = i000.end();
    auto i456b = i456.begin();
    auto i456e = i456.end();

    using except_t = std::runtime_error;

    smooth_type defaulted_corr;
    smooth_type scalar_corr{};
    smooth_type tensor_corr{4, 5, 6};

    SECTION("defaulted") {
        REQUIRE(shapes[0].slice(eb, ee, eb, ee) == defaulted_corr);
    }

    SECTION("Scalar") {
        REQUIRE(shapes[1].slice(eb, ee, eb, ee) == scalar_corr);
    }

    SECTION("Vector") {
        REQUIRE(shapes[2].slice(i0b, i0e, i2b, i2e) == smooth_type{2});
    }

    SECTION("matrix") {
        REQUIRE(shapes[3].slice(i11b, i11e, i32b, i32e) == smooth_type{2, 1});
    }

    SECTION("tensor") {
        REQUIRE(shapes[4].slice(i000b, i000e, i456b, i456e) == tensor_corr);
    }

    SECTION("Different size ranges") {
        REQUIRE_THROWS_AS(shapes[0].slice(i0b, i0e, eb, ee), except_t);
        REQUIRE_THROWS_AS(shapes[1].slice(i0b, i0e, eb, ee), except_t);

        // Catch it in first preliminary check
        REQUIRE_THROWS_AS(shapes[2].slice(i0b, i0e, eb, ee), except_t);

        // // Catch it in the loop
        REQUIRE_THROWS_AS(shapes[3].slice(i11b, i11e, i2b, i2e), except_t);

        // Catch it after the loop
        REQUIRE_THROWS_AS(shapes[4].slice(i0b, i0e, i11b, i11e), except_t);
    }

    SECTION("Last element < first element") {
        REQUIRE_THROWS_AS(shapes[3].slice(i2b, i2e, i0b, i0e), except_t);
    }
}

} // namespace

TEST_CASE("SmoothCommon", "shape") {
    using smooth_type = tensorwrapper::shape::Smooth;
    using smooth_view = tensorwrapper::shape::SmoothView<smooth_type>;
    using const_view  = tensorwrapper::shape::SmoothView<const smooth_type>;

    std::vector<smooth_type> shapes;
    smooth_type defaulted;
    shapes.push_back(defaulted);
    shapes.push_back(smooth_type{});
    shapes.push_back(smooth_type{3});
    shapes.push_back(smooth_type{2, 4});
    shapes.push_back(smooth_type{4, 5, 6});

    std::vector<smooth_view> shape_views;
    std::vector<const_view> const_shape_views;
    for(std::size_t i = 0; i < shapes.size(); ++i) {
        shape_views.push_back(smooth_view(shapes[i]));
        const_shape_views.push_back(const_view(shapes[i]));
    }

    SECTION("extent") {
        test_extent(shapes);
        test_extent(shape_views);
        test_extent(const_shape_views);
    }

    SECTION("slice(initializer_lists)") {
        test_slice_il(shapes);
        test_slice_il(shape_views);
        test_extent(const_shape_views);
    }

    SECTION("slice(containers)") {
        test_slice_container(shapes);
        test_slice_container(shape_views);
        test_slice_container(const_shape_views);
    }

    SECTION("slice(ranges)") {
        test_slice_ranges(shapes);
        test_slice_ranges(shape_views);
        test_slice_ranges(const_shape_views);
    }
}
