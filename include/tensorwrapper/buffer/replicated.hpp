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
#include <tensorwrapper/buffer/local.hpp>
#include <tensorwrapper/buffer/replicated_common.hpp>

namespace tensorwrapper::buffer {

/** @brief Denotes that a buffer is the same on all processes.
 *
 *  At the moment this class is a strong type and has no additional state over
 *  its base class.
 */
class Replicated : public ReplicatedCommon<Replicated>, public Local {
private:
    /// Type *this derives from
    using my_base_type = ReplicatedCommon<Replicated>;

public:
    // Pull in base's ctors
    using Local::Local;

protected:
    friend my_base_type;

    virtual const_element_reference get_elem_(index_vector index) const = 0;
    virtual void set_elem_(index_vector index, element_type value)      = 0;
};

} // namespace tensorwrapper::buffer
