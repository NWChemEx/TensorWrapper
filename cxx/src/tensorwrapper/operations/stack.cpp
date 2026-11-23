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

#include <algorithm>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/operations/stack.hpp>

namespace tensorwrapper::operations {
namespace {

/* Copies one input tensor into its slot of the stacked output.
 *
 * In row-major order the input is @p outer contiguous blocks of @p inner
 * elements. Block o of input k lands at block (o * n_tensors + k) of the
 * output.
 */
class StackKernel {
public:
    using size_type = std::size_t;

    StackKernel(buffer::Contiguous& out, size_type k, size_type n_tensors,
                size_type outer, size_type inner) :
      m_pout_(&out),
      m_k_(k),
      m_n_tensors_(n_tensors),
      m_outer_(outer),
      m_inner_(inner) {}

    template<typename FloatType>
    void operator()(std::span<FloatType> in) const {
        using clean_type = std::remove_const_t<FloatType>;
        // Throws if the output (i.e., tensor 0) holds a different type
        auto out = buffer::get_raw_data<clean_type>(*m_pout_);
        for(size_type o = 0; o < m_outer_; ++o) {
            auto in_begin = in.begin() + o * m_inner_;
            auto offset   = (o * m_n_tensors_ + m_k_) * m_inner_;
            std::copy(in_begin, in_begin + m_inner_, out.begin() + offset);
        }
    }

private:
    buffer::Contiguous* m_pout_;
    size_type m_k_;
    size_type m_n_tensors_;
    size_type m_outer_;
    size_type m_inner_;
};

} // namespace

Tensor stack(const std::vector<Tensor>& tensors, int axis) {
    using size_type = std::size_t;

    if(tensors.empty())
        throw std::runtime_error("stack: need at least one tensor to stack");

    const auto& buffer0 = buffer::make_contiguous(tensors.front().buffer());
    const auto shape0   = buffer0.shape();
    for(const auto& t : tensors) {
        if(buffer::make_contiguous(t.buffer()).shape() != shape0)
            throw std::runtime_error(
              "stack: all input tensors must have the same shape");
    }

    // N.b. rank is the rank of the input tensors. The output tensor has
    // rank + 1. axis will be the rank of the output tensor.
    const int rank = static_cast<int>(shape0.rank());
    if(axis < -(rank + 1) || axis > rank)
        throw std::runtime_error("stack: axis is out of range");
    const auto uaxis =
      static_cast<size_type>(axis < 0 ? axis + rank + 1 : axis);

    const size_type n_tensors = tensors.size();
    std::vector<size_type> extents;
    size_type outer = 1;
    size_type inner = 1;
    for(size_type i = 0; i < shape0.rank(); ++i) {
        if(i == uaxis) extents.push_back(n_tensors);
        extents.push_back(shape0.extent(i));
        if(i < uaxis)
            outer *= shape0.extent(i);
        else
            inner *= shape0.extent(i);
    }
    if(uaxis == shape0.rank()) extents.push_back(n_tensors);

    shape::Smooth out_shape(extents.begin(), extents.end());
    auto out_buffer = buffer::make_contiguous(buffer0, out_shape);

    for(size_type k = 0; k < n_tensors; ++k) {
        StackKernel kernel(out_buffer, k, n_tensors, outer, inner);
        const auto& buffer_k = buffer::make_contiguous(tensors[k].buffer());
        buffer::visit_contiguous_buffer(kernel, buffer_k);
    }

    return Tensor(std::move(out_shape), std::move(out_buffer));
}

} // namespace tensorwrapper::operations
