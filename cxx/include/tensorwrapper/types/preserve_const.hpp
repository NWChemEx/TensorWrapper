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

#pragma once
#include <type_traits>
namespace tensorwrapper::types {

/**@ brief Uses the const-ness of @p T to ensure @p U has the same const-ness.
 *
 *  @tparam T The type to preserve the const-ness of.
 *  @tparam U The type to preserve the const-ness of @p T on.
 */
template<typename T, typename U = T>
using preserve_const_t = std::conditional_t<std::is_const_v<T>, const U, U>;

} // namespace tensorwrapper::types
