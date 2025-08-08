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
#include <tensorwrapper/allocator/allocator.hpp>
#include <tensorwrapper/buffer/buffer.hpp>
#include <tensorwrapper/layout/layout.hpp>
#include <tensorwrapper/shape/shape.hpp>
#include <tensorwrapper/utilities/block_diagonal_matrix.hpp>
#include <tensorwrapper/utilities/floating_point_dispatch.hpp>

namespace tensorwrapper::utilities {

namespace {

struct BlockDiagonalMatrixKernel {
    template<typename FloatType>
    auto run(const buffer::BufferBase& b, const std::vector<Tensor>& matrices) {
        using allocator_type = tensorwrapper::allocator::Eigen<FloatType>;

        // All inputs must be Rank 2, square, and the same floating point type.
        // If so, sum their extent sizes.
        std::size_t size = 0;
        for(const auto& matrix : matrices) {
            if(!allocator_type::can_rebind(matrix.buffer()))
                throw std::runtime_error(
                  "All inputs must have the same floating point type");

            if(matrix.rank() != 2)
                throw std::runtime_error(
                  "All inputs must be matrices (Rank == 2)");

            const auto& mshape = matrix.buffer().layout().shape().as_smooth();
            if(mshape.extent(0) != mshape.extent(1))
                throw std::runtime_error("All inputs must be square matrices");

            size += mshape.extent(0);
        }

        // Allocate new buffer
        allocator_type allocator(b.allocator().runtime());
        shape::Smooth oshape{size, size};
        layout::Physical olayout(oshape);
        auto obuffer = allocator.construct(olayout, 0.0);

        // Copy values from input into corresponding blocks
        std::size_t offset = 0;
        for(const auto& matrix : matrices) {
            const auto& mbuffer = allocator.rebind(matrix.buffer());
            auto extent = mbuffer.layout().shape().as_smooth().extent(0);
            for(std::size_t i = 0; i < extent; ++i) {
                for(std::size_t j = 0; j < extent; ++j) {
                    obuffer->set_elem({offset + i, offset + j},
                                      mbuffer.get_elem({i, j}));
                }
            }
            offset += extent;
        }
        return Tensor(oshape, std::move(obuffer));
    }
};

} // namespace

Tensor block_diagonal_matrix(std::vector<Tensor> matrices) {
    const auto& buffer0 = matrices[0].buffer();
    BlockDiagonalMatrixKernel kernel;
    return floating_point_dispatch(kernel, buffer0, matrices);
}

} // namespace tensorwrapper::utilities
