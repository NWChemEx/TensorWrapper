#pragma once
#include "tensorwrapper/detail_/hashing.hpp"
#include "tensorwrapper/tensor/fields.hpp"
#include "tensorwrapper/tensor/shapes/shape.hpp"
#include <memory>
#include <string>
#include <type_traits>

namespace tensorwrapper::tensor::buffer {
namespace detail_ {
template<typename FieldType>
class BufferPIMPL;

}

/** @brief Wraps a tensor backend.
 *
 *  The Buffer class provides a uniform API for accessing the underlying tensor
 *  library, depending only on whether the underlying tensor has scalar elements
 *  or tensor elements.
 *
 *  @tparam FieldType The type of the field the tensor is over. Expected to be
 *                    either field::Scalar or field::Tensor.
 */
template<typename FieldType>
class Buffer {
private:
    /// Type of the PIMPL
    using pimpl_type = detail_::BufferPIMPL<FieldType>;

    /// Trait for determining if the fields are the same
    template<typename T>
    static constexpr bool same_field_v = std::is_same_v<FieldType, T>;

    /// Enables a function, via SFINAE, if @p T is different than FieldType
    template<typename T, typename U = void>
    using enable_if_diff_field_t = std::enable_if_t<!same_field_v<T>, U>;

    using my_type = Buffer<FieldType>;

public:
    /// Type used for indices in einstein/index-based operations
    using annotation_type = std::string;

    /// Type of a read-only reference to an annotation
    using const_annotation_reference = const std::string&;

    /// Type of a pointer to the PIMPL
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /// Type used to model the shape
    using shape_type = Shape<FieldType>;

    /// Type of a read-only reference to the shape
    using const_shape_reference = const shape_type&;

    /// Type of the object used for hashing
    using hasher_type = tensorwrapper::detail_::Hasher;

    /// Mutable reference to a hasher
    using hasher_reference = hasher_type&;

    /** @brief Defaulted default ctor.
     *
     *  This ctor creates an uninitialized Buffer instance. The resulting
     *  instance has no PIMPL and can only be used after assigning an
     *  initialized Buffer instance to it.
     *
     *  @throw None No throw guarantee.
     */
    Buffer() noexcept;

    /** @brief PIMPL Ctor.
     *
     *  Generally speaking Buffer instances are created by Allocator instances.
     *  The allocator instances are implemented in the source and have access to
     *  the backend specific PIMPLs (e.g. TABufferPIMPL). Thus what will usually
     *  happen is that the allocator makes a backend specific PIMPL and then
     *  uses that PIMPL to initialize a Buffer instance via this ctor.
     *
     *  @param[in] pimpl An initialized backend specific PIMPL, passed via the
     *                   base class.
     *
     *  @throw None No throw guarantee.
     */

    Buffer(pimpl_pointer pimpl) noexcept;

    Buffer(const Buffer& other);

    Buffer(Buffer&& other) noexcept;

    Buffer& operator=(const Buffer& rhs);

    Buffer& operator=(Buffer&& rhs) noexcept;

    ~Buffer() noexcept;

    /** @brief Exchanges the state of this Buffer with that of @p other.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Buffer& other) noexcept;

    /** @brief Used to determine if the Buffer wraps an actual tensor or not.
     *
     *  @return true if the Buffer is currently wrapping a tensor and false
     *          otherwise.
     *
     *  @throw None No throw gurantee.
     */
    bool is_initialized() const noexcept;

    /** @brief Scales (and optionally permutes) a tensor
     *
     *  This function scales a tensor using einstein notation. This means that
     *  if the indices on the left side of the equation are not in the same
     *  order as those on the right, in addition to scaling the tensor, this
     *  function will permute the modes.
     *
     *  ```.cpp
     *  // To run B("i,j") = 4.2 * A("j,i") run:
     *  A.scale("j,i", "i,j", B, 4.2);
     *  ```
     *
     *  @param[in] my_idx The einstein indices for the current buffer.
     *  @param[in] out_idx The einstein indices for the returned buffer.
     *  @param[in,out] out The buffer to put the result into. If @p out was
     *                     default initialized, this function will default
     *                     initiailze a PIMPL of the same type as the current
     *                     instance before assigning to it.
     *  @param[in] rhs The value to scale this tensor by.
     *
     *  @throws std::runtime_error if the present buffer is not initialized.
     *                             Strong throw guarantee.
     */
    void scale(const_annotation_reference my_idx,
               const_annotation_reference out_idx, my_type& out,
               double rhs) const;

    /** @brief Adds (and optionally permutes) two tensors.
     *
     *  This function adds two tensors together, obtaining a third tensor. The
     *  addition is specified using einstein notation, which also allows one to
     *  permute the modes of the tensor as part of the operation.
     *
     *  ```.cpp
     *  // To run C("i,j") = A("j,i") + B("i,j") run:
     *  A.add("j,i", "i,j", C, "i,j", B);
     *  ```
     *
     *  @param[in] my_idx The einstein indices for the current buffer.
     *  @param[in] out_idx The einstein indices for the output buffer.
     *  @param[in,out] out The buffer to put the answer in. If @p out is default
     *                     initialized, this function will default initialize an
     *                     instance of this buffer's PIMPL before assigning to
     *                     @p out.
     *  @param[in] rhs_idx The einstein indices for the buffer being added to
     *                     this buffer.
     *  @param[in] rhs     The buffer being added to this buffer.
     *
     *  @throw std::runtime_error if the present buffer is not initialized.
     *                            Strong throw guarantee.
     *  @throw std::runtime_error if @p rhs is not initialize. Strong throw
     *                            gurantee.
     */
    void add(const_annotation_reference my_idx,
             const_annotation_reference out_idx, my_type& out,
             const_annotation_reference rhs_idx, const my_type& rhs) const;

    void inplace_add(const_annotation_reference my_idx,
                     const_annotation_reference rhs_idx, const my_type& rhs);

    /** @brief Subtracts (and optionally permutes) two tensors.
     *
     *  This function subtracts two tensors together, obtaining a third tensor.
     *  The subtraction is specified using einstein notation, which also allows
     *  one to permute the modes of the tensor as part of the operation.
     *
     *  ```.cpp
     *  // To run C("i,j") = A("j,i") - B("i,j") run:
     *  A.subtract("j,i", "i,j", C, "i,j", B);
     *  ```
     *
     *  @param[in] my_idx The einstein indices for the current buffer.
     *  @param[in] out_idx The einstein indices for the output buffer.
     *  @param[in,out] out The buffer to put the answer in. If @p out is default
     *                     initialized, this function will default initialize an
     *                     instance of this buffer's PIMPL before assigning to
     *                     @p out.
     *  @param[in] rhs_idx The einstein indices for the buffer being added to
     *                     this buffer.
     *  @param[in] rhs     The buffer being added to this buffer.
     *
     *  @throw std::runtime_error if the present buffer is not initialized.
     *                            Strong throw guarantee.
     *  @throw std::runtime_error if @p rhs is not initialize. Strong throw
     *                            gurantee.
     */
    void subtract(const_annotation_reference my_idx,
                  const_annotation_reference out_idx, my_type& out,
                  const_annotation_reference rhs_idx, const my_type& rhs) const;

    void inplace_subtract(const_annotation_reference my_idx,
                          const_annotation_reference rhs_idx,
                          const my_type& rhs);

    /** @brief Multiplies (and optionally permutes) two tensors.
     *
     *  This function multiplies two tensors together, obtaining a third tensor.
     *  The multiplication is specified using einstein notation, which also
     *  allows one to perform contractions and/or permute the modes of the
     *  tensor as part of the operation.
     *
     *  ```.cpp
     *  // To run C("i,j") = A("j,i") * B("i,j") run:
     *  A.times("j,i", "i,j", C, "i,j", B);
     *  ```
     *
     *  @param[in] my_idx The einstein indices for the current buffer.
     *  @param[in] out_idx The einstein indices for the output buffer.
     *  @param[in,out] out The buffer to put the answer in. If @p out is default
     *                     initialized, this function will default initialize an
     *                     instance of this buffer's PIMPL before assigning to
     *                     @p out.
     *  @param[in] rhs_idx The einstein indices for the buffer being added to
     *                     this buffer.
     *  @param[in] rhs     The buffer being added to this buffer.
     *
     *  @throw std::runtime_error if the present buffer is not initialized.
     *                            Strong throw guarantee.
     *  @throw std::runtime_error if @p rhs is not initialize. Strong throw
     *                            gurantee.
     */
    void times(const_annotation_reference my_idx,
               const_annotation_reference out_idx, my_type& out,
               const_annotation_reference rhs_idx, const my_type& rhs) const;

    bool operator==(const Buffer& rhs) const noexcept;

    /** @brief Compares two buffers with different fields.
     *
     *  @tparam T The field for @p rhs. This function only participates in
     *            overload resolution if @p T is different than FieldType.
     *  @tparam <anonymous> Template parameter used to disable this function,
     *                      via SFINAE, when @p T is the same as FieldType.
     *
     *  @param[in] rhs The buffer to compare to.
     *
     *  @return False buffers are always different if they use different fields.
     *
     *  @throw None No throw guarantee.
     */
    template<typename T, typename = enable_if_diff_field_t<T>>
    bool operator==(const Buffer<T>& rhs) const noexcept {
        return false;
    }

private:
    /// Asserts the PIMPL is initialized, throwing std::runtime_error if not
    void assert_initialized_() const;

    /// Initializes @p other 's PIMPL with a default version of this's PIMPL
    void default_initialize_(Buffer& other) const;

    /// Actually stores the state and implements the tensor.
    pimpl_pointer m_pimpl_;
};

/** @brief Determines if two buffers are different.
 *
 *  @relates Buffer
 *
 *  This function simply negates operator== and thus relies on its definition of
 *  equality.
 *
 *  @tparam T The field type for the buffer to the left of operator!=
 *  @tparam U The field type for the buffer to the right of operator!=
 *
 *  @param[in] lhs The buffer on the left
 *  @param[in] rhs The buffer on the right.
 *
 *  @return False if @p lhs is equal to @p rhs and true otherwise.
 *
 *  @throw None No throw guarantee.
 */
template<typename T, typename U>
bool operator!=(const Buffer<T>& lhs, const Buffer<U>& rhs) {
    return !(lhs == rhs);
}

extern template class Buffer<field::Scalar>;
extern template class Buffer<field::Tensor>;

} // namespace tensorwrapper::tensor::buffer
