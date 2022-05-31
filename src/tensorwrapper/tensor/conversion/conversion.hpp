#pragma once
#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include "tensorwrapper/tensor/buffer/buffer.hpp"

namespace tensorwrapper::tensor::conversion {

template<typename ToType>
struct Conversion;

template<typename TileType>
struct Conversion<TA::DistArray<TileType, TA::SparsePolicy>> {
    /// The type of the output tensor
    using output_t = TA::DistArray<TileType, TA::SparsePolicy>;

    /// The type of Buffer
    template<typename FieldType>
    using buffer_t = buffer::Buffer<FieldType>;

    /// The implementation of a buffer storing a TA Tensor
    template<typename FieldType>
    using ta_pimpl_t = buffer::detail_::TABufferPIMPL<FieldType>;

    template<typename FieldType>
    output_t& convert(buffer_t<FieldType>& B) {
        /// Try to cast the input buffer's PIMPL to a TABufferPIMPL
        auto ta_buffer_pimpl = dynamic_cast<ta_pimpl_t<FieldType>*>(B.pimpl());
        if(ta_buffer_pimpl == nullptr) { throw std::bad_cast(); }

        /// Return the Tensor stored in the PIMPL
        return std::get<output_t>(ta_buffer_pimpl->m_tensor_);
    }
};

} // namespace tensorwrapper::tensor::conversion