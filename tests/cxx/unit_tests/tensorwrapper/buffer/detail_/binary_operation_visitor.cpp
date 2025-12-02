/*
 * Copyright 2025 NWChemEx-Project
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

#include "../../testing/testing.hpp"
#include <tensorwrapper/buffer/detail_/binary_operation_visitor.hpp>
#include <tensorwrapper/types/floating_point.hpp>
using namespace tensorwrapper;

/* Testing notes:
 *
 * In testing the derived classes we assume that the backends have been
 * exhaustively tested. Therefore, we simply ensure that each overload works
 * correctly and that the correct backend is dispatched to.
 */
TEMPLATE_LIST_TEST_CASE("BinaryOperationVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::BinaryOperationVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    buffer_type this_buffer(std::vector<TestType>(6, TestType(0.0)));

    label_type this_labels("i,j");
    shape_type this_shape({2, 3});

    label_type lhs_labels("i,k");
    shape_type lhs_shape({2, 4});

    label_type rhs_labels("k,j");
    shape_type rhs_shape({4, 3});

    VisitorType visitor(this_buffer, this_labels, this_shape, lhs_labels,
                        lhs_shape, rhs_labels, rhs_shape);

    REQUIRE(visitor.this_shape() == this_shape);
    REQUIRE(visitor.lhs_shape() == lhs_shape);
    REQUIRE(visitor.rhs_shape() == rhs_shape);

    REQUIRE(visitor.this_labels() == this_labels);
    REQUIRE(visitor.lhs_labels() == lhs_labels);
    REQUIRE(visitor.rhs_labels() == rhs_labels);

    std::span<const double> dspan;
    std::span<const float> fspan;
    REQUIRE_THROWS_AS(visitor(dspan, fspan), std::runtime_error);
}

TEMPLATE_LIST_TEST_CASE("AdditionVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::AdditionVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    TestType one{1.0}, two{2.0}, three{3.0}, four{4.0};
    std::vector<TestType> this_data{one, two, three, four};
    std::vector<TestType> lhs_data{four, three, two, one};
    std::vector<TestType> rhs_data{one, one, one, one};
    shape_type shape({4});
    label_type labels("i");

    std::span<TestType> lhs_span(lhs_data.data(), lhs_data.size());
    std::span<const TestType> clhs_span(lhs_data.data(), lhs_data.size());
    std::span<TestType> rhs_span(rhs_data.data(), rhs_data.size());
    std::span<const TestType> crhs_span(rhs_data.data(), rhs_data.size());

    SECTION("existing buffer") {
        buffer_type this_buffer(this_data);
        VisitorType visitor(this_buffer, labels, shape, labels, shape, labels,
                            shape);

        visitor(lhs_span, rhs_span);
        REQUIRE(this_buffer.at(0) == TestType(5.0));
        REQUIRE(this_buffer.at(1) == TestType(4.0));
        REQUIRE(this_buffer.at(2) == TestType(3.0));
        REQUIRE(this_buffer.at(3) == TestType(2.0));
    }

    SECTION("non-existing buffer") {
        buffer_type empty_buffer;
        VisitorType visitor(empty_buffer, labels, shape, labels, shape, labels,
                            shape);

        visitor(clhs_span, crhs_span);
        REQUIRE(empty_buffer.at(0) == TestType(5.0));
        REQUIRE(empty_buffer.at(1) == TestType(4.0));
        REQUIRE(empty_buffer.at(2) == TestType(3.0));
        REQUIRE(empty_buffer.at(3) == TestType(2.0));
    }
}

TEMPLATE_LIST_TEST_CASE("SubtractionVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::SubtractionVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    TestType one{1.0}, two{2.0}, three{3.0}, four{4.0};
    std::vector<TestType> this_data{one, two, three, four};
    std::vector<TestType> lhs_data{four, three, two, one};
    std::vector<TestType> rhs_data{one, one, one, one};
    shape_type shape({4});
    label_type labels("i");

    std::span<TestType> lhs_span(lhs_data.data(), lhs_data.size());
    std::span<const TestType> clhs_span(lhs_data.data(), lhs_data.size());
    std::span<TestType> rhs_span(rhs_data.data(), rhs_data.size());
    std::span<const TestType> crhs_span(rhs_data.data(), rhs_data.size());

    SECTION("existing buffer") {
        buffer_type this_buffer(this_data);
        VisitorType visitor(this_buffer, labels, shape, labels, shape, labels,
                            shape);

        visitor(lhs_span, rhs_span);
        REQUIRE(this_buffer.at(0) == TestType(3.0));
        REQUIRE(this_buffer.at(1) == TestType(2.0));
        REQUIRE(this_buffer.at(2) == TestType(1.0));
        REQUIRE(this_buffer.at(3) == TestType(0.0));
    }

    SECTION("non-existing buffer") {
        buffer_type empty_buffer;
        VisitorType visitor(empty_buffer, labels, shape, labels, shape, labels,
                            shape);

        visitor(clhs_span, crhs_span);
        REQUIRE(empty_buffer.at(0) == TestType(3.0));
        REQUIRE(empty_buffer.at(1) == TestType(2.0));
        REQUIRE(empty_buffer.at(2) == TestType(1.0));
        REQUIRE(empty_buffer.at(3) == TestType(0.0));
    }
}
