#pragma once
#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include "tensorwrapper/tensor/buffer/buffer.hpp"
#include <TiledArray/dist_array.h>
#include <TiledArray/tensor.h>

namespace tensorwrapper::tensor {

/** @brief  Dispatches to the correct conversion function based on the
 *          desired output type of the wrapped tensor.
 *
 *  This is the primary template for the Converion; it is not defined. This
 *  template is chosen when no specialization exists for the requested
 *  operation, resulting in a compiler error.
 *
 *  @tparam ToType The type of the output tensor.
 */
template<typename ToType>
struct Conversion;

/** @brief  Specialization for DistArray conversion cases.
 *
 *  This specialization is intended to return the tensor wrapped by a Buffer
 *  back as a TA DistArray. Since we only have the TA backend at the moment,
 *  this just grabs a reference to the tensor that the buffer is holding onto.
 *
 *  @tparam TileType The type of the tiles in the DistArray.
 */
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

    /** @brief Checks if this Conversion instance can convert @p B.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param B The buffer we want to convert
     *
     *  @return True if convertable, false
     */
    template<typename FieldType>
    bool can_convert(buffer_t<FieldType>& B) {
        /// If cast fails, then can't convert
        auto ta_buffer_pimpl = dynamic_cast<ta_pimpl_t<FieldType>*>(B.pimpl());
        if(ta_buffer_pimpl == nullptr) { return false; }

        /// Check output_t against field traits
        return field_traits::template is_tensor_type_v<output_t>;
    }

    /** @brief Return the tensor wrapped in @p B as a TA DistArray.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param B The buffer we want to convert
     *
     *  @return The wrapped TA DistArray
     */
    template<typename FieldType>
    output_t& convert(buffer_t<FieldType>& B) {
        /// Try to cast the input buffer's PIMPL to a TABufferPIMPL
        auto ta_buffer_pimpl = dynamic_cast<ta_pimpl_t<FieldType>*>(B.pimpl());
        if(ta_buffer_pimpl == nullptr) { throw std::bad_cast(); }

        /// Return the Tensor stored in the PIMPL
        return std::get<output_t>(ta_buffer_pimpl->m_tensor_);
    }
};

/// Typedef of a Conversion instance which can make TA::TSpArrayD instances
using to_ta_distarrayd_t = Conversion<TA::TSpArrayD>;

using to_ta_totd_t =
  Conversion<TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>>;
} // namespace tensorwrapper::tensor
