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
#include <type_traits>

namespace tensorwrapper::buffer::detail_ {

/** @brief Dispatches to the appropriate backend based on the FP type.
 *
 *
 *
 */
class BinaryOperationVisitor {
public:
    using buffer_type      = wtf::buffer::FloatBuffer;
    using string_type      = std::string;
    using label_type       = dsl::DummyIndices<string_type>;
    using shape_type       = shape::Smooth;
    using const_shape_view = shape::SmoothView<const shape_type>;

    BinaryOperationVisitor(buffer_type& this_buffer, label_type this_labels,
                           shape_type this_shape, label_type lhs_labels,
                           shape_type lhs_shape, label_type rhs_labels,
                           shape_type rhs_shape) :
      m_pthis_buffer_(&this_buffer),
      m_this_labels_(std::move(this_labels)),
      m_this_shape_(std::move(this_shape)),
      m_lhs_labels_(std::move(lhs_labels)),
      m_lhs_shape_(std::move(lhs_shape)),
      m_rhs_labels_(std::move(rhs_labels)),
      m_rhs_shape_(std::move(rhs_shape)) {}

    const auto& this_shape() const { return m_this_shape_; }
    const auto& lhs_shape() const { return m_lhs_shape_; }
    const auto& rhs_shape() const { return m_rhs_shape_; }

    const auto& this_labels() const { return m_this_labels_; }
    const auto& lhs_labels() const { return m_lhs_labels_; }
    const auto& rhs_labels() const { return m_rhs_labels_; }

    template<typename LHSType, typename RHSType>
        requires(!std::is_same_v<LHSType, RHSType>)
    void operator()(std::span<const LHSType>, std::span<const RHSType>) {
        throw std::runtime_error(
          "BinaryOperationVisitor: Mixed types not supported");
    }

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
    auto make_lhs_eigen_tensor_(std::span<const FloatType> data) {
        /// XXX: Ideally we would not need to const_cast here, but we didn't
        ///      code EigenTensor correctly...

        auto* pdata = const_cast<FloatType*>(data.data());
        std::span<FloatType> non_const_data(pdata, data.size());
        return backends::eigen::make_eigen_tensor(non_const_data, m_lhs_shape_);
    }

    template<typename FloatType>
    auto make_rhs_eigen_tensor_(std::span<const FloatType> data) {
        /// XXX: Ideally we would not need to const_cast here, but we didn't
        ///      code EigenTensor correctly...

        auto* pdata = const_cast<FloatType*>(data.data());
        std::span<FloatType> non_const_data(pdata, data.size());
        return backends::eigen::make_eigen_tensor(non_const_data, m_rhs_shape_);
    }

private:
    buffer_type* m_pthis_buffer_;
    label_type m_this_labels_;
    shape_type m_this_shape_;

    label_type m_lhs_labels_;
    shape_type m_lhs_shape_;

    label_type m_rhs_labels_;
    shape_type m_rhs_shape_;
};

class AdditionVisitor : public BinaryOperationVisitor {
public:
    using BinaryOperationVisitor::BinaryOperationVisitor;

    // AdditionVisitor(shape, permutation, shape, permutation)
    template<typename FloatType>
    void operator()(std::span<const FloatType> lhs,
                    std::span<const FloatType> rhs) {
        using clean_t = std::decay_t<FloatType>;
        auto pthis    = this->make_this_eigen_tensor_<clean_t>();
        auto plhs     = this->make_lhs_eigen_tensor_(lhs);
        auto prhs     = this->make_rhs_eigen_tensor_(rhs);

        pthis->addition_assignment(this_labels(), lhs_labels(), rhs_labels(),
                                   *plhs, *prhs);
    }
};

} // namespace tensorwrapper::buffer::detail_
