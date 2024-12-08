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

#pragma once
#include "detail_/eigen_dispatcher.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor {

/** @brief Converts tensors to Eigen::tensor then executes the operation.
 *
 *
 */
class Eigen {
public:
    using labeled_tensor = typename tensorwrapper::Tensor::labeled_tensor_type;

    static labeled_tensor assign(labeled_tensor lhs, labeled_tensor rhs) {
        return lhs;
    }

    static labeled_tensor add(labeled_tensor result, labeled_tensor lhs,
                              labeled_tensor rhs) {
        return rhs;
    }
};

} // namespace tensorwrapper::dsl::executor