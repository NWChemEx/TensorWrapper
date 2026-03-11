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
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/buffer/buffer_view_base.hpp>
#include <tensorwrapper/types/preserve_const.hpp>
namespace tensorwrapper::buffer {

/** @brief Establishes that the state in the buffer is obtainable without
 *         communication.
 *
 *  For now this class is a strong type and does not impart any additional state
 *  to the BufferBase class.
 *
 */
class Local : public BufferBase {
private:
    /// Type *this inherits from
    using my_base_type = BufferBase;

public:
    // Pull in base's ctors
    using my_base_type::my_base_type;
};

/** @brief A view of a Local buffer.
 *
 *  This class is a view of a Local buffer. It is used to create a view of a
 *  Local buffer. It is not a strong type and does not impart any additional
 *  state to the BufferViewBase class.
 */
template<typename LocalType>
class LocalView
  : public BufferViewBase<types::preserve_const_t<LocalType, BufferBase>> {
private:
    using buffer_base_type = types::preserve_const_t<LocalType, BufferBase>;
    using my_base_type     = BufferViewBase<buffer_base_type>;

public:
    /// Pull in base's ctors
    using my_base_type::my_base_type;
};

} // namespace tensorwrapper::buffer
