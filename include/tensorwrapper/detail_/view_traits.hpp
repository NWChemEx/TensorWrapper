#pragma once
#include <type_traits>

namespace tensorwrapper::detail_ {

/** @brief Is the cast from @p FromType to @p ToType just adding const?
 *
 *  A common TMP pattern in implementing views is needing to convert mutable
 *  views to read-only views. This trait can be used to compare the template
 *  type parameters of two views (assuming the views are templated on what
 *  object they are acting like) in order to determine if they represent a
 *  conversion from @p FromType to @p ToType such that @p ToType is
 *  `const FromType`. If @p ToType is `const FromType` this template variable
 *  will be set to true, otherwise it will be set to false.
 *
 *  @tparam FromType The type we are converting from.
 *  @tparam ToType The type we are converting to.
 */
template<typename FromType, typename ToType>
constexpr bool is_mutable_to_immutable_cast_v =
  !std::is_const_v<FromType> &&           // FromType is NOT read-only
  std::is_const_v<ToType> &&              // ToType is read-only
  std::is_same_v<const FromType, ToType>; // They differ by const-ness

/** @brief Disables a templated function except when
 *         `is_mutable_to_immutable_cast_v<FromType, ToType>` evaluates to true.
 *
 *  If `View` is a template class with template parameter type `T`, we want the
 *  implicit conversion from `View<T>` to `View<const T>` to exist. In practice,
 *  this leaves us with two options: partial specialization of `View` for
 *  const-qualified types or use of SFINAE to disable the conversion. We prefer
 *  the latter as the former requires us to duplicate the entirety of the
 *  class. This template type will disable the accompanying function via SFINAE
 *  if @p ToType is not `const FromType`.
 *
 *  @tparam FromType The type we are converting from. Expected to be the
 *                   template type parameter of the view we are casting from.
 *  @tparam ToType The type we are converting to. Expected to be the template
 *                 type parameter of the view we are casting to.
 */
template<typename FromType, typename ToType>
using enable_if_mutable_to_immutable_cast_t =
  std::enable_if_t<is_mutable_to_immutable_cast_v<FromType, ToType>>;

} // namespace tensorwrapper::detail_