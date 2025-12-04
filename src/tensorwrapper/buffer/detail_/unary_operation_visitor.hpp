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
#include "../../backends/eigen/eigen_tensor_impl.hpp"
#include <span>
#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/shape/smooth_view.hpp>
#include <type_traits>
#include <wtf/wtf.hpp>

namespace tensorwrapper::buffer::detail_ {

/** @brief Dispatches to the appropriate backend based on the FP type.
 *
 *  This visitor is intended to be used with WTF's buffer visitation mechanism.
 *  This base class implements the logic common to all unary operations and
 *  lets the derived classes implement the operation-specific logic.
 *
 */
class UnaryOperationVisitor {
public:
    /// Type of the WTF buffer
    using buffer_type = wtf::buffer::FloatBuffer;

    /// Type that the labels use for representing indices
    using string_type = std::string;

    /// Type of a set of labels
    using label_type = dsl::DummyIndices<string_type>;

    /// Type describing the shape of the tensors
    using shape_type = shape::Smooth;

    /// Type describing a read-only view acting like shape_type
    using const_shape_view = shape::SmoothView<const shape_type>;

    UnaryOperationVisitor(buffer_type& this_buffer, label_type this_labels,
                          shape_type this_shape, label_type other_labels,
                          shape_type other_shape) :
      m_pthis_buffer_(&this_buffer),
      m_this_labels_(std::move(this_labels)),
      m_this_shape_(std::move(this_shape)),
      m_other_labels_(std::move(other_labels)),
      m_other_shape_(std::move(other_shape)) {}

    const auto& this_shape() const { return m_this_shape_; }
    const auto& other_shape() const { return m_other_shape_; }

    const auto& this_labels() const { return m_this_labels_; }
    const auto& other_labels() const { return m_other_labels_; }

protected:
    template<typename FloatType>
    auto make_eigen_tensor_(std::span<FloatType> data, const_shape_view shape) {
        return backends::eigen::make_eigen_tensor(data, shape);
    }

    template<typename FloatType>
    auto make_this_eigen_tensor_() {
        if(m_pthis_buffer_->size() != m_this_shape_.size()) {
            std::vector<FloatType> temp_buffer(m_this_shape_.size());
            *m_pthis_buffer_ = buffer_type(std::move(temp_buffer));
        }
        auto this_span =
          wtf::buffer::contiguous_buffer_cast<FloatType>(*m_pthis_buffer_);
        return backends::eigen::make_eigen_tensor(this_span, m_this_shape_);
    }

    template<typename FloatType>
    auto make_other_eigen_tensor_(std::span<FloatType> data) {
        /// XXX: Ideally we would not need to const_cast here, but we didn't
        ///      code EigenTensor correctly...

        using clean_type = std::decay_t<FloatType>;
        auto* pdata      = const_cast<clean_type*>(data.data());
        std::span<clean_type> non_const_data(pdata, data.size());
        return backends::eigen::make_eigen_tensor(non_const_data,
                                                  m_other_shape_);
    }

private:
    buffer_type* m_pthis_buffer_;
    label_type m_this_labels_;
    shape_type m_this_shape_;

    label_type m_other_labels_;
    shape_type m_other_shape_;
};

class PermuteVisitor : public UnaryOperationVisitor {
public:
    using UnaryOperationVisitor::UnaryOperationVisitor;

    template<typename FloatType>
    void operator()(std::span<FloatType> other) {
        using clean_t = std::decay_t<FloatType>;
        auto pthis    = this->make_this_eigen_tensor_<clean_t>();
        auto pother   = this->make_other_eigen_tensor_(other);

        pthis->permute_assignment(this->this_labels(), other_labels(), *pother);
    }
};

class ScalarMultiplicationVisitor : public UnaryOperationVisitor {
public:
    using scalar_type = wtf::fp::Float;
    ScalarMultiplicationVisitor(buffer_type& this_buffer,
                                label_type this_labels, shape_type this_shape,
                                label_type other_labels, shape_type other_shape,
                                scalar_type scalar) :
      UnaryOperationVisitor(this_buffer, this_labels, this_shape, other_labels,
                            other_shape),
      m_scalar_(scalar) {}

    template<typename FloatType>
    void operator()(std::span<FloatType> other) {
        using clean_t = std::decay_t<FloatType>;
        auto pthis    = this->make_this_eigen_tensor_<clean_t>();
        auto pother   = this->make_other_eigen_tensor_(other);

        // TODO: Change when public API changes to support other FP types
        auto scalar = wtf::fp::float_cast<double>(m_scalar_);
        pthis->scalar_multiplication(this->this_labels(), other_labels(),
                                     scalar, *pother);
    }

private:
    scalar_type m_scalar_;
};

} // namespace tensorwrapper::buffer::detail_
