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
#include <tensorwrapper/buffer/detail_/unary_operation_visitor.hpp>
#include <tensorwrapper/types/floating_point.hpp>
using namespace tensorwrapper;

/* Testing notes:
 *
 * In testing the derived classes we assume that the backends have been
 * exhaustively tested. Therefore, we simply ensure that each overload works
 * correctly and that the correct backend is dispatched to.
 */
TEMPLATE_LIST_TEST_CASE("UnaryOperationVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::UnaryOperationVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    buffer_type this_buffer(std::vector<TestType>(6, TestType(0.0)));

    label_type this_labels("i,j");
    shape_type this_shape({2, 3});

    label_type other_labels("i,k");
    shape_type other_shape({2, 4});

    VisitorType visitor(this_buffer, this_labels, this_shape, other_labels,
                        other_shape);

    REQUIRE(visitor.this_shape() == this_shape);
    REQUIRE(visitor.other_shape() == other_shape);

    REQUIRE(visitor.this_labels() == this_labels);
    REQUIRE(visitor.other_labels() == other_labels);
}

TEMPLATE_LIST_TEST_CASE("PermuteVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::PermuteVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    label_type this_labels("i,j");
    shape_type this_shape({2, 3});

    label_type other_labels("j,i");
    shape_type other_shape({3, 2});

    std::vector<TestType> other_data = {TestType(1.0), TestType(2.0),
                                        TestType(3.0), TestType(4.0),
                                        TestType(5.0), TestType(6.0)};
    std::span<TestType> other_span(other_data.data(), other_data.size());
    std::span<const TestType> cother_span(other_data.data(), other_data.size());

    SECTION("Buffer is allocated") {
        buffer_type this_buffer(std::vector<TestType>(6, TestType(0.0)));
        VisitorType visitor(this_buffer, this_labels, this_shape, other_labels,
                            other_shape);
        visitor(other_span);

        REQUIRE(this_buffer.at(0) == TestType(1.0));
        REQUIRE(this_buffer.at(1) == TestType(3.0));
        REQUIRE(this_buffer.at(2) == TestType(5.0));
        REQUIRE(this_buffer.at(3) == TestType(2.0));
        REQUIRE(this_buffer.at(4) == TestType(4.0));
        REQUIRE(this_buffer.at(5) == TestType(6.0));
    }

    SECTION("Buffer is not allocated") {
        buffer_type this_buffer;
        VisitorType visitor(this_buffer, this_labels, this_shape, other_labels,
                            other_shape);
        visitor(cother_span);

        REQUIRE(this_buffer.at(0) == TestType(1.0));
        REQUIRE(this_buffer.at(1) == TestType(3.0));
        REQUIRE(this_buffer.at(2) == TestType(5.0));
        REQUIRE(this_buffer.at(3) == TestType(2.0));
        REQUIRE(this_buffer.at(4) == TestType(4.0));
        REQUIRE(this_buffer.at(5) == TestType(6.0));
    }
}

TEMPLATE_LIST_TEST_CASE("ScalarMultiplicationVisitor", "[buffer][detail_]",
                        types::floating_point_types) {
    using VisitorType = buffer::detail_::ScalarMultiplicationVisitor;
    using buffer_type = typename VisitorType::buffer_type;
    using label_type  = typename VisitorType::label_type;
    using shape_type  = typename VisitorType::shape_type;

    label_type this_labels("i,j");
    shape_type this_shape({2, 3});

    label_type other_labels("j,i");
    shape_type other_shape({3, 2});

    std::vector<TestType> other_data = {TestType(1.0), TestType(2.0),
                                        TestType(3.0), TestType(4.0),
                                        TestType(5.0), TestType(6.0)};
    std::span<TestType> other_span(other_data.data(), other_data.size());
    std::span<const TestType> cother_span(other_data.data(), other_data.size());

    // TODO: when public API of MDBuffer supports other FP types, test them here
    double scalar_{2.0};
    TestType scalar(scalar_);

    SECTION("Buffer is allocated") {
        buffer_type this_buffer(std::vector<TestType>(6, TestType(0.0)));
        VisitorType visitor(this_buffer, this_labels, this_shape, other_labels,
                            other_shape, scalar_);
        visitor(other_span);

        REQUIRE(this_buffer.at(0) == TestType(1.0) * scalar);
        REQUIRE(this_buffer.at(1) == TestType(3.0) * scalar);
        REQUIRE(this_buffer.at(2) == TestType(5.0) * scalar);
        REQUIRE(this_buffer.at(3) == TestType(2.0) * scalar);
        REQUIRE(this_buffer.at(4) == TestType(4.0) * scalar);
        REQUIRE(this_buffer.at(5) == TestType(6.0) * scalar);
    }

    SECTION("Buffer is not allocated") {
        buffer_type this_buffer;
        VisitorType visitor(this_buffer, this_labels, this_shape, other_labels,
                            other_shape, scalar_);
        visitor(cother_span);

        REQUIRE(this_buffer.at(0) == TestType(1.0) * scalar);
        REQUIRE(this_buffer.at(1) == TestType(3.0) * scalar);
        REQUIRE(this_buffer.at(2) == TestType(5.0) * scalar);
        REQUIRE(this_buffer.at(3) == TestType(2.0) * scalar);
        REQUIRE(this_buffer.at(4) == TestType(4.0) * scalar);
        REQUIRE(this_buffer.at(5) == TestType(6.0) * scalar);
    }
}
