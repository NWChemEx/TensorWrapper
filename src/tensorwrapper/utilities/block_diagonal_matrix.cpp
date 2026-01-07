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
#include <tensorwrapper/buffer/buffer.hpp>
#include <tensorwrapper/layout/layout.hpp>
#include <tensorwrapper/shape/shape.hpp>
#include <tensorwrapper/utilities/block_diagonal_matrix.hpp>

namespace tensorwrapper::utilities {

namespace {

struct Initializer {
    explicit Initializer(shape::Smooth shape) : m_shape(std::move(shape)) {}

    template<typename FloatType>
    void operator()(const std::span<FloatType>) {
        using clean_type = std::decay_t<FloatType>;
        m_buffer         = buffer::make_contiguous<clean_type>(m_shape);
    }

    buffer::Contiguous m_buffer;
    shape::Smooth m_shape;
};

struct BlockDiagonalMatrixKernel {
    // Initializes assuming square matrix
    BlockDiagonalMatrixKernel(buffer::Contiguous& buffer, std::size_t offset,
                              std::size_t extent) :
      m_pbuffer(&buffer),
      m_offset(offset),
      m_row_extent(extent),
      m_col_extent(extent) {}

    template<typename FloatType>
    void operator()(const std::span<FloatType> matrix_i) {
        for(std::size_t i = 0; i < m_row_extent; ++i) {
            for(std::size_t j = 0; j < m_col_extent; ++j) {
                m_pbuffer->set_elem({m_offset + i, m_offset + j},
                                    matrix_i[i * m_col_extent + j]);
            }
        }
    }

    buffer::Contiguous* m_pbuffer;

    std::size_t m_offset;

    std::size_t m_row_extent;
    std::size_t m_col_extent;
};

} // namespace

Tensor block_diagonal_matrix(std::vector<Tensor> matrices) {
    if(matrices.empty()) {
        Tensor t;
        return t; // No idea why the compiler won't let us do 'return {};' here
    }

    // All inputs must be Rank 2, square, and the same floating point type.
    // If so, sum their extent sizes.
    std::size_t size = 0;
    std::vector<std::size_t> row_extents(matrices.size());
    for(const auto& matrix : matrices) {
        if(matrix.rank() != 2)
            throw std::runtime_error("All inputs must be matrices (Rank == 2)");

        const auto& mshape = matrix.buffer().layout().shape().as_smooth();
        if(mshape.extent(0) != mshape.extent(1))
            throw std::runtime_error("All inputs must be square matrices");

        row_extents.push_back(mshape.extent(0));
        size += row_extents.back();
    }

    shape::Smooth shape{size, size};
    layout::Physical olayout(shape);

    Initializer init_kernel(shape);
    const auto& buffer0 = buffer::make_contiguous(matrices.front().buffer());
    buffer::visit_contiguous_buffer(init_kernel, buffer0);

    buffer::Contiguous buffer = std::move(init_kernel.m_buffer);

    std::size_t offset = 0;

    for(const auto& matrix : matrices) {
        const auto& buffer_i   = buffer::make_contiguous(matrix.buffer());
        std::size_t row_extent = buffer_i.shape().extent(0);
        BlockDiagonalMatrixKernel kernel(buffer, offset, row_extent);
        buffer::visit_contiguous_buffer(kernel, buffer_i);
        offset += row_extent;
    }
    layout::Logical llayout(shape);
    return Tensor(std::move(buffer), std::move(llayout), std::move(olayout));
}

} // namespace tensorwrapper::utilities
