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

// #include <tensorwrapper/buffer/detail_/addition_visitor.hpp>
// #include <tensorwrapper/types/floating_point.hpp>

// using namespace tensorwrapper;

// TEMPLATE_LIST_TEST_CASE("AdditionVisitor", "[buffer][detail_]",
//                         types::floating_point_types) {
//     using VisitorType = buffer::detail_::AdditionVisitor;

//     VisitorType visitor;

//     SECTION("vectors") {
//         std::vector<TestType> lhs{1.0, 2.0, 3.0};
//         std::vector<TestType> rhs{4.0, 5.0, 6.0};

//         visitor(std::span<TestType>(lhs), std::span<const TestType>(rhs));

//         REQUIRE(lhs[0] == Approx(5.0).epsilon(1e-10));
//         REQUIRE(lhs[1] == Approx(7.0).epsilon(1e-10));
//         REQUIRE(lhs[2] == Approx(9.0).epsilon(1e-10));
//     }
// }
