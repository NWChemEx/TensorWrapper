#pragma once
#include <cassert>
#include <cstddef>

namespace tensorwrapper::detail_ {

/** @brief Safely converts objects to std::size_t.
 *
 *  @tparam The integral type of the input.
 *
 *  This function will ensure at compile time that @tparam T is an integral type
 *  and will assert that it is greater than equal to 0 at runtime.
 *
 *  @note assert is used, instead of throw, so that the overhead for the checks
 *        can be disabled in Release mode. Given that this function is used in
 *        the getting/setting of tensor elements by offsets, its overhead could
 *        conceivably add up.
 *
 *  @param[in] i The integer to convert to `std::size_t`
 *
 *  @return @p i cast to a `std::size_t`
 */
template<typename T>
std::size_t to_size_t(T i) {
    static_assert(std::is_integral_v<std::decay_t<T>>);
    assert(i >= 0);
    return i;
}

/** @brief Safely converts integral objects to long.
 *
 *  @tparam The type of @p i. Must be an integral type.
 *
 *  @param[in] i The integer we are converting.
 *
 *  @note See the note on to_size_t for details on bounds checking.
 *
 *  @return @p i cast to a `long`.
 */
template<typename T>
long to_long(T i) {
    static_assert(std::is_integral_v<std::decay_t<T>>);
    if constexpr(std::is_same_v<T, std::size_t>) {
        assert(i < std::numeric_limits<long>::max());
    }
    return i;
}

} // namespace tensorwrapper::detail_
