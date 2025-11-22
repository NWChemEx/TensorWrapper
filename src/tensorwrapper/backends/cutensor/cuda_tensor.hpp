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
#include <span>
#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/shape/smooth_view.hpp>

namespace tensorwrapper::backends::cutensor {

/** @brief Wraps using cuTENSOR
 *
 *  @tparam FloatType Floating point type used for the tensor's elements
 *
 *  N.b. The name of this class is chosen to avoid conflict with cuTENSOR.
 */
template<typename FloatType>
class CUDATensor {
private:
    /// Type of *this
    using my_type = CUDATensor<FloatType>;

    /// Read-only reference to an object of type my_type
    using const_my_reference = const my_type&;

public:
    using value_type       = FloatType;
    using span_type        = std::span<value_type>;
    using shape_type       = shape::Smooth;
    using const_shape_view = shape::SmoothView<const shape_type>;
    using label_type       = dsl::DummyIndices<std::string>;
    using size_type        = std::size_t;

    CUDATensor(span_type data, const_shape_view shape) :
      m_data_(data), m_shape_(shape) {}

    void contraction_assignment(label_type this_labels, label_type lhs_labels,
                                label_type rhs_labels, const_my_reference lhs,
                                const_my_reference rhs);

    size_type rank() const noexcept { return m_shape_.rank(); }

    size_type size() const noexcept { return m_shape_.size(); }

    auto shape() const noexcept { return m_shape_; }

    auto data() noexcept { return m_data_.data(); }

    auto data() const noexcept { return m_data_.data(); }

private:
    span_type m_data_;

    const_shape_view m_shape_;
};

extern template class CUDATensor<float>;
extern template class CUDATensor<double>;

} // namespace tensorwrapper::backends::cutensor
