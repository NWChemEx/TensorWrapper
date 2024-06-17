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
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper::detail_ {

class TensorPIMPL {
public:
    /// Type *this implements
    using parent_type = Tensor;

    /// Type of a pointer to *this
    using pimpl_pointer = typename parent_type::pimpl_pointer;
};

} // namespace tensorwrapper::detail_
