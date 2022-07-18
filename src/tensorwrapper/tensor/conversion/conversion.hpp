#pragma once
#include "../buffer/detail_/buffer_pimpl.hpp"
#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include "../detail_/ta_traits.hpp"
#include <TiledArray/dist_array.h>
#include <TiledArray/tensor.h>
#include <tensorwrapper/tensor/buffer/buffer.hpp>
#include <utilities/type_traits/variant/has_type.hpp>

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

    /// The type of Buffer
    template<typename FieldType>
    using buffer_pimpl_t = buffer::detail_::BufferPIMPL<FieldType>;

    /// The implementation of a buffer storing a TA Tensor
    template<typename FieldType>
    using ta_pimpl_t = buffer::detail_::TABufferPIMPL<FieldType>;

    /** @brief Checks if this Conversion instance can convert @p Bp.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param Bp The BufferPIMPL we want to convert
     *
     *  @return True if convertable, false
     */
    template<typename FieldType>
    bool can_convert(const buffer_pimpl_t<FieldType>& Bp) {
        /// If cast fails, then can't convert
        auto ta_buffer_pimpl = dynamic_cast<ta_pimpl_t<FieldType>*>(Bp);
        if(ta_buffer_pimpl == nullptr) { return false; }

        /// Check output_t against field traits
        using ta_traits = detail_::TiledArrayTraits<FieldType>;
        using variant_t = typename ta_traits::variant_type;
        return utilities::type_traits::variant::has_type_v<output_t, variant_t>;
    }

    /** @brief Checks if this Conversion instance can convert @p B.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param B The buffer we want to convert
     *
     *  @return True if convertable, false
     */
    template<typename FieldType>
    bool can_convert(const buffer_t<FieldType>& B) {
        /// If cast fails, then can't convert
        return can_convert(B.pimpl());
    }

    /** @brief Return the tensor wrapped in @p Bp as a TA DistArray.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param Bp The BufferPIMPL we want to convert
     *
     *  @return The wrapped TA DistArray
     */
    template<typename FieldType>
    output_t& convert(buffer_pimpl_t<FieldType>& Bp) {
        /// Try to cast the input buffer's PIMPL to a TABufferPIMPL
        auto ta_buffer_pimpl = dynamic_cast<ta_pimpl_t<FieldType>*>(Bp);
        if(ta_buffer_pimpl == nullptr) { throw std::bad_cast(); }

        /// Return the Tensor stored in the PIMPL
        return std::get<output_t>(ta_buffer_pimpl->m_tensor_);
    }

    /** @brief Return the tensor wrapped in @p B as a TA DistArray.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param B The Buffer we want to convert
     *
     *  @return The wrapped TA DistArray
     */
    template<typename FieldType>
    output_t& convert(buffer_t<FieldType>& B) {
        return convert(B.pimpl());
    }

    /** @brief Return the tensor wrapped in @p Bp as a const TA DistArray.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param Bp The BufferPIMPL we want to convert
     *
     *  @return The wrapped TA DistArray
     */
    template<typename FieldType>
    const output_t& convert(const buffer_pimpl_t<FieldType>& Bp) {
        /// Try to cast the input buffer's PIMPL to a TABufferPIMPL
        auto ta_buffer_pimpl = dynamic_cast<const ta_pimpl_t<FieldType>*>(Bp);
        if(ta_buffer_pimpl == nullptr) { throw std::bad_cast(); }

        /// Return the Tensor stored in the PIMPL
        return std::get<const output_t>(ta_buffer_pimpl->m_tensor_);
    }

    /** @brief Return the tensor wrapped in @p B as a const TA DistArray.
     *
     *  @tparam FieldType The type of value in the Buffer
     *
     *  @param B The Buffer we want to convert
     *
     *  @return The wrapped TA DistArray
     */
    template<typename FieldType>
    const output_t& convert(const buffer_t<FieldType>& B) {
        return convert(B.pimpl());
    }
};

/// Typedef of a Conversion instance which can make TA::TSpArrayD instances
using to_ta_distarrayd_t = Conversion<TA::TSpArrayD>;

using to_ta_totd_t =
  Conversion<TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>>;
} // namespace tensorwrapper::tensor
