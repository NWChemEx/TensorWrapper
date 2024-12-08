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
#include <tensorwrapper/allocator/eigen.hpp>
#include <utilities/type_traits/variant/cat.hpp>

namespace tensorwrapper::allocator::detail_ {

/** @brief Unwraps a type-erased EigenBuffer object into a std::variant.
 *
 *  Eigen has templated their tensor class on the floating point type and the
 *  rank of the tensor. The latter is particularly annoying to deal with because
 *  TensorWrapper treats the rank as a runtime value. To get around this we
 *  define a variant type, EigenBufferUnwrapper::variant_type, that is capable
 *  of holding every EigenBuffer object from rank 0 to rank
 *  "EigenBufferUnwrapper::max_rank". The TMP for working out the typedef
 *  "EigneBufferUnwrapper::variant_type" lives within *this. The remainder of
 *  *this contains a static method `downcast` which takes care of the downcast
 *  into the variant.
 *
 */
struct EigenBufferUnwrapper {
    /// The maximum Eigen rank we are considering
    static constexpr std::size_t max_rank = 10;

    /// Type giving us the numbers [0, max_rank) in a parameter pack
    using sequence_type = std::make_index_sequence<max_rank>;

    /// Type of rank @p I buffer::Eigen object with @p FloatType elements
    template<typename FloatType, std::size_t I>
    using buffer_type = buffer::Eigen<FloatType, I>;

    /// Used only to work out the variant_type for @p FloatType
    template<typename FloatType, std::size_t... I>
    static auto dummy(std::index_sequence<I...>)
      -> std::variant<buffer_type<FloatType, I>...>;

    /// Concatenates variants with floating point types @p Args
    template<typename... Args>
    using variant_type_ =
      utilities::type_traits::variant::cat_t<decltype(dummy<Args>(
        std::declval<sequence_type>()))...>;

    /// The variant for the floating points we are considering
    using variant_type = variant_type_<float, double>;

    /** @brief API for downcasting a buffer and putting it into a variant.
     *
     *  @tparam BufferType The base type of the buffer we are downcasting.
     *                     BufferType may be a reference and/or cv-qualified.
     *
     *  This method is a user-friendly API for the downcast. It wraps the
     *  less friendly API which requires explicit template parameters.
     *
     *  @param[in] buffer The object to downcast.
     *
     *  @return A std::variant containing the downcast buffer.
     *
     *  @throw std::runtime_error if the rank of the tensor exceeds `max_rank`.
     *                            Strong throw guarantee.
     *  @throw std::runtime_error if the buffer contains a floating-point type
     *                            *this does not support. See the definition of
     *                            `variant_type` for the supported floating-
     *                            point types. Strong throw guarantee.
     */
    template<typename BufferType>
    static variant_type downcast(BufferType&& buffer) {
        return downcast_<0>(std::forward<BufferType>(buffer));
    }

private:
    /** @brief Tries to downcast to EigenBuffer<T, Rank> for all known Ts.
     *
     *  This method implements downcast by recursion. At each level of recursion
     *  we compare the runtime rank of `buffer` to the compile time rank of the
     *  function (given by `Rank`). If they match we do the downcast; if they
     *  don't match we recurse. The downcasts explicitly try each floating point
     *  type we support.
     *  @throw std::runtime_error if the rank of the tensor exceeds `max_rank`.
     *                            Strong throw guarantee.
     *  @throw std::runtime_error if the buffer contains a floating-point type
     *                            *this does not support. See the definition of
     *                            `variant_type` for the supported floating-
     *                            point types. Strong throw guarantee.
     *
     */
    template<unsigned short Rank, typename BufferType>
    static variant_type downcast_(BufferType&& buffer) {
        // Prevents infinite recursion and provides a runtime error if the
        // runtime rank of the tensor is too high.
        if constexpr(Rank == max_rank) {
            throw std::runtime_error("Please increase max_rank");
        } else {
            // Does buffer have rank "Rank"?
            if(buffer.layout().shape().rank() == Rank) {
                using eigend_type = buffer_type<double, Rank>;
                using eigenf_type = buffer_type<float, Rank>;

                auto pdouble_buffer = dynamic_cast<eigend_type*>(&buffer);
                if(pdouble_buffer != nullptr)
                    return variant_type(*pdouble_buffer);
                else {
                    auto pfloat_buffer = dynamic_cast<eigenf_type*>(&buffer);
                    if(pfloat_buffer == nullptr)
                        throw std::runtime_error("Not convertible.");
                    return variant_type{*pfloat_buffer};
                }
            }

            // Recurse and try the next rank
            return downcast_<Rank + 1>(std::forward<BufferType>(buffer));
        }
    }
};

} // namespace tensorwrapper::allocator::detail_