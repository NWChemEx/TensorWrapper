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

#include "executor/executor.hpp"
#include <tensorwrapper/dsl/parser.hpp>

namespace tensorwrapper::dsl {

// Specialize this class to set the class based on the objects being combined
template<typename ObjectType>
struct ExecutorType;

// Sets the default executor for tensors
template<>
struct ExecutorType<Tensor> {
    using executor_type = executor::Eigen;
};

// Typedef to shorten retrieving the type of default executor for @p ObjectType
template<typename ObjectType>
using default_executor_type = typename ExecutorType<ObjectType>::executor_type;

#define TPARAMS template<typename ObjectType, typename LabelType>
#define PARSER Parser<ObjectType, LabelType>
#define LABELED_TYPE typename PARSER::labeled_type

TPARAMS LABELED_TYPE PARSER::assign(labeled_type lhs, labeled_type rhs) {
    return default_executor_type<ObjectType>::assign(std::move(lhs),
                                                     std::move(rhs));
}

TPARAMS LABELED_TYPE PARSER::add(labeled_type result, labeled_type lhs,
                                 labeled_type rhs) {
    return default_executor_type<ObjectType>::add(
      std::move(result), std::move(lhs), std::move(rhs));
}

#undef PARSER
#undef TPARAMS

template class Parser<Tensor, std::string>;

} // namespace tensorwrapper::dsl