// /*
//  * Copyright 2024 NWChemEx-Project
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #include <stdexcept>
// #include <tensorwrapper/dsl/pairwise_parser.hpp>
// #include <tensorwrapper/tensor/tensor_class.hpp>

// namespace tensorwrapper::dsl {
// namespace {
// struct CallAddition {
//     template<typename LHSType, typename RHSType>
//     static decltype(auto) run(LHSType&& lhs, RHSType&& rhs) {
//         const auto& llabels = lhs.rhs();
//         return lhs.lhs().addition(llabels, std::forward<RHSType>(rhs));
//     }
// };

// template<typename FunctorType, typename ResultType, typename LHSType,
//          typename RHSType>
// decltype(auto) binary_op(ResultType&& result, LHSType&& lhs, RHSType&& rhs) {
//     auto& rv_object        = result.lhs();
//     const auto& lhs_object = lhs.lhs();
//     const auto& rhs_object = rhs.lhs();

//     const auto& lhs_labels = lhs.rhs();
//     const auto& rhs_labels = rhs.rhs();

//     using object_type = typename std::decay_t<ResultType>::object_type;

//     if constexpr(std::is_same_v<object_type, Tensor>) {
//         if(rv_object == Tensor{}) {
//             const auto& llayout = lhs_object.logical_layout();
//             // const auto& rlayout = rhs_object.logical_layout();
//             std::decay_t<decltype(llayout)> rv_layout(
//               llayout); // FunctorType::run(llayout(lhs_labels),
//                         // rlayout(rhs_labels));

//             auto lbuffer = lhs_object.buffer()(lhs_labels);
//             auto rbuffer = rhs_object.buffer()(rhs_labels);
//             auto buffer  = FunctorType::run(lbuffer, rbuffer);

//             // TODO figure out permutation
//             Tensor(std::move(rv_layout), std::move(buffer)).swap(rv_object);
//         } else {
//             throw std::runtime_error("Hints are not allowed yet!");
//         }
//     } else {
//         // Getting here means the assert will fail
//         static_assert(std::is_same_v<object_type, Tensor>, "NYI");
//     }
//     return result;
// }
// } // namespace

// #define TPARAMS template<typename ObjectType, typename LabelType>
// #define PARSER PairwiseParser<ObjectType, LabelType>
// #define LABELED_TYPE typename PARSER::labeled_type

// TPARAMS LABELED_TYPE PARSER::add(labeled_type result, labeled_type lhs,
//                                  labeled_type rhs) {
//     return binary_op<CallAddition>(result, lhs, rhs);
// }

// #undef PARSER
// #undef TPARAMS

// template class PairwiseParser<Tensor, std::string>;

// } // namespace tensorwrapper::dsl