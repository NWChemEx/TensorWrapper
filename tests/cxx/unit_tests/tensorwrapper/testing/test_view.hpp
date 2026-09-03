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
#include <stdexcept>
#include <tensorwrapper/buffer/buffer_view_base.hpp>

namespace tensorwrapper::testing {

/// BufferViewBase is abstract (get_elem_()/set_elem_() are pure virtual), so
/// it can not be instantiated directly. This minimal derived class exists so
/// tests that only need BufferViewBase's own layout/construction/equality
/// behavior (and not real element access) can still get a concrete object.
/// get_elem_()/set_elem_() just throw, mirroring what an unsupported view
/// would do.
template<typename BufferBaseType>
class TestView : public buffer::BufferViewBase<BufferBaseType> {
private:
    using base_type = buffer::BufferViewBase<BufferBaseType>;

public:
    using base_type::base_type;

protected:
    using typename base_type::const_element_reference;
    using typename base_type::element_type;
    using typename base_type::index_vector;

    const_element_reference get_elem_(index_vector) const override {
        throw std::runtime_error("TestView does not implement get_elem_");
    }

    void set_elem_(index_vector, element_type) override {
        throw std::runtime_error("TestView does not implement set_elem_");
    }
};

} // namespace tensorwrapper::testing
