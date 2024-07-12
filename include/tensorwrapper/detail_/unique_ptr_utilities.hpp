#pragma once
#include <memory>

namespace tensorwrapper::detail_ {

/** @brief Implements a dynamic cast of a unique_ptr.
 *
 *  @tparam T The object type to cast from.
 *  @tparam U The object type to cast to.
 *
 *  The C++ standard library does not implement dynamic cast for unique pointers
 *  (because there is no way to do this without two variables thinking they own
 *  the memory). This function implements dynamic cast by essentially swapping
 *  the raw pointers in two unique pointers (one of which is a nullptr) when the
 *  object pointed to by  @p pbase can be dynamically casted to @p U. This
 *  minimizes the time when the single owner violation occurs (and encapsulates
 *  it to this function).
 *
 *  @param[in,out] pbase The pointer we are dynamic casting. If the cast
 *                       succeeds @p pbase will be set to the nullptr. If the
 *                       cast fails @p pbase will be unchanged.
 *
 *  @return If the cast succeeds a new `std::unique_ptr<U>` object which
 *          owns the dynamic casted memory originally owned by @p pbase.
 *          Otherwise a nullptr.
 *
 *  @throw None No throw guarantee.
 */
template<typename U, typename T>
std::unique_ptr<U> dynamic_pointer_cast(std::unique_ptr<T>& pbase) {
    auto pderived_raw = dynamic_cast<U*>(pbase.get());
    if(pderived_raw) pbase.release();
    return std::unique_ptr<U>(pderived_raw);
}

} // namespace tensorwrapper::detail_
