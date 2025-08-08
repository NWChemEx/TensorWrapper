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

#pragma once

namespace tensorwrapper::types {

/** @brief Defines the member types for the @p ClassType  class.
 *
 *  This class will serve as the single-source of truth for defining the member
 *  types for the @p ClassType class. The primary template is not defined and
 *  developers are expected to specialize the template for each @p ClassType
 *  in the TensorWrapper library.
 *
 *  @tparam ClassType The, possibly cv-qualified, type of the class which *this
 *                    defines the types for.
 */
template<typename ClassType>
struct ClassTraits;

} // namespace tensorwrapper::types
