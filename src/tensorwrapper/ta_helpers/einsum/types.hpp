/*
 * Copyright 2022 NWChemEx-Project
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
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tensorwrapper::ta_helpers::einsum::types {

using size        = std::size_t;
using range       = std::pair<size, size>;
using index       = std::string;
using index_set   = std::vector<index>;
using assoc_index = std::map<index, size>;
using assoc_range = std::map<index, range>;

} // namespace tensorwrapper::ta_helpers::einsum::types