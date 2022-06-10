#pragma once
#include <type_traits>

/** @file fields.hpp
 *
 *  Informally speaking, a field is a set of things for which addition,
 *  multiplication, and their inverses are defined. For our purposes, what we
 *  care about are the things in the field (the elements of the field) as these
 *  are what we can define our tensor components in terms of.
 *
 *  Practically we have two types of tensors:
 *  - Tensors whose elements are numbers (either real or complex)
 *  - Tensors whose elements are tensors of the first kind
 *
 *  In theory, we could continue the nesting (*e.g.*, have tensors whose
 *  elements are tensors of the second kind), but we only support single
 *  nesting as that's all TiledArray supports.
 *
 *  Design wise the fields are (right now) just strong types. Under the hood,
 *  each field maps to a std::variant of possible tile types. The use of the
 *  fields avoids having the variants appear in the TensorWrapper types, i.e.,
 *  instead of:
 *
 *  @code
 *
 *  TensorWrapper<std::variant<TA::Tensor<float>, TA::Tensor<double>, ...>;
 *
 *  @endcode
 *
 *  we get:
 *
 *  @code
 *  TensorWrapper<Scalar>;
 *  @endcode
 */

namespace tensorwrapper::tensor::field {

/// Represents a field containing real or complex numbers
struct Scalar {};

/// Represents a field containing tensors
struct Tensor {};

template<typename FieldType>
inline constexpr bool is_scalar_field_v = std::is_same_v<FieldType, Scalar>;

template<typename FieldType>
inline constexpr bool is_tensor_field_v = std::is_same_v<FieldType, Tensor>;

template<typename FieldType, typename U = void>
struct enable_if_scalar_field {
    using type = typename std::enable_if<is_scalar_field_v<FieldType>, U>::type;
};

template<typename FieldType, typename U = void>
struct enable_if_tensor_field {
    using type = typename std::enable_if<is_tensor_field_v<FieldType>, U>::type;
};

template<typename FieldType, typename U = void>
using enable_if_scalar_field_t =
  typename enable_if_scalar_field<FieldType, U>::type;
template<typename FieldType, typename U = void>
using enable_if_tensor_field_t =
  typename enable_if_tensor_field<FieldType, U>::type;

} // namespace tensorwrapper::tensor::field