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
#include <stdexcept>
#include <tensorwrapper/buffer/buffer_fwd.hpp>
#include <tensorwrapper/types/buffer_traits.hpp>
#include <tuple>
#include <utility>

namespace tensorwrapper::buffer {

/** @brief CRTP base factoring the common layout/equality API of BufferBase and
 *         BufferViewBase.
 *
 *  Derived must implement the protected hooks: has_layout_(), layout_(),
 *  and approximately_equal_(const BufferBase&, double).
 *
 *  @tparam Derived The CRTP derived type (BufferBase or BufferViewBase).
 */
template<typename Derived>
class BufferBaseCommon {
private:
    /// Type of *this
    using my_type = BufferBaseCommon<Derived>;

    /// Traits for Derived
    using traits_type = types::ClassTraits<Derived>;

public:
    ///@{
    using layout_type            = typename traits_type::layout_type;
    using layout_reference       = typename traits_type::layout_reference;
    using layout_pointer         = typename traits_type::layout_pointer;
    using const_layout_reference = typename traits_type::const_layout_reference;
    using rank_type              = typename traits_type::rank_type;
    using size_type              = typename traits_type::size_type;
    using element_type           = typename traits_type::element_type;
    using const_element_reference =
      typename traits_type::const_element_reference;
    using index_vector = typename traits_type::index_vector;
    ///@}

    // -------------------------------------------------------------------------
    // -- Accessors
    // -------------------------------------------------------------------------

    /** @brief Does *this have a layout?
     *
     *  @return True if *this has a layout and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool has_layout() const noexcept { return derived_().has_layout_(); }

    /** @brief Retrieves the layout of *this.
     *
     *  @return A reference to the layout.
     *
     *  @throw std::runtime_error if *this does not have a layout. Strong throw
     *                            guarantee.
     */
    layout_reference layout() {
        assert_layout_();
        return derived_().layout_();
    }

    /** @brief Retrieves the layout of *this.
     *
     *  @return A read-only reference to the layout.
     *
     *  @throw std::runtime_error if *this does not have a layout. Strong throw
     *                            guarantee.
     */
    const_layout_reference layout() const {
        assert_layout_();
        return derived_().layout_();
    }

    /** @brief Returns the rank of the layout.
     *
     *  @return The rank, or 0 if *this has no layout.
     *
     *  @throw None No throw guarantee.
     */
    rank_type rank() const noexcept {
        return has_layout() ? layout().rank() : 0;
    }

    /** @brief Returns the element with the offsets specified by @p index.
     *
     *  @param[in] index The offsets into each mode of *this for the desired
     *                   element. The length of @p index must equal rank().
     *
     *  @return A const reference to the element at the specified offsets.
     *
     *  @throw std::out_of_range if the length of @p index does not equal
     *                           rank() or if any entry in @p index is out of
     *                           bounds. Strong throw guarantee.
     *  @throw std::runtime_error if *this does not support element access
     *                            (e.g. *this is a view with no such
     *                            implementation). Strong throw guarantee.
     */
    const_element_reference get_element(index_vector index) const {
        return derived_().get_elem_(std::move(index));
    }

    /** @brief Sets the element with the offsets specified by @p index to
     *         @p value.
     *
     *  @param[in] index The offsets into each mode of *this for the desired
     *                   element. The length of @p index must equal rank().
     *  @param[in] value The new value for the specified element.
     *
     *  @throw std::out_of_range if the length of @p index does not equal
     *                           rank() or if any entry in @p index is out of
     *                           bounds. Strong throw guarantee.
     *  @throw std::runtime_error if *this does not support element access
     *                            (e.g. *this is a view with no such
     *                            implementation). Strong throw guarantee.
     */
    void set_element(index_vector index, element_type value) {
        derived_().set_elem_(std::move(index), std::move(value));
    }

    /** @brief Returns the element with the offsets given by @p offsets.
     *
     *  This overload allows the offsets to be provided as an arbitrary
     *  number of arguments (one per mode) instead of as an index_vector. It
     *  is implemented in terms of get_element(index_vector) const.
     *
     *  @tparam Offsets The types of the offsets. Expected to be integral
     *                  types implicitly convertible to size_type.
     *
     *  @param[in] offsets The offsets into each mode of *this for the
     *                     desired element.
     *
     *  @return A const reference to the element at the specified offsets.
     *
     *  @throw std::out_of_range if the number of offsets does not equal
     *                           rank() or if any offset is out of bounds.
     *                           Strong throw guarantee.
     */
    template<typename... Offsets>
    const_element_reference get_element(Offsets... offsets) const {
        return get_element(index_vector{static_cast<size_type>(offsets)...});
    }

    /** @brief Sets the element with the offsets given by @p offsets to
     *         @p value.
     *
     *  This overload allows the offsets to be provided as an arbitrary
     *  number of arguments (one per mode) instead of as an index_vector. It
     *  is implemented in terms of set_element(index_vector, element_type).
     *
     *  @tparam Offsets The types of the offsets. Expected to be integral
     *                  types implicitly convertible to size_type.
     *
     *  @param[in] offsets The offsets into each mode of *this for the
     *                     desired element.
     *  @param[in] value The new value for the specified element.
     *
     *  @throw std::out_of_range if the number of offsets does not equal
     *                           rank() or if any offset is out of bounds.
     *                           Strong throw guarantee.
     */
    template<typename... Args>
    void set_element(Args... args) {
        static_assert(sizeof...(Args) >= 1,
                      "set_element requires at least a value argument");
        set_element_unpack_(std::make_index_sequence<sizeof...(Args) - 1>{},
                            std::make_tuple(args...));
    }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this value equal to @p rhs?
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename OtherDerived>
    bool operator==(const BufferBaseCommon<OtherDerived>& rhs) const noexcept {
        if(has_layout() != rhs.has_layout()) return false;
        if(has_layout() && layout().are_different(rhs.layout())) return false;
        return true;
    }

    /** @brief Is *this different from @p rhs?
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename OtherDerived>
    bool operator!=(const BufferBaseCommon<OtherDerived>& rhs) const noexcept {
        return !(*this == rhs);
    }

    /** @brief Are *this and @p rhs approximately equal within @p tol?
     *
     *  @param[in] rhs The object to compare to.
     *  @param[in] tol The tolerance for the comparison.
     *
     *  @return True if approximately equal, false otherwise.
     */
    template<typename OtherDerived>
    bool approximately_equal(const BufferBaseCommon<OtherDerived>& rhs,
                             double tol) const {
        return derived_().approximately_equal_(rhs.derived_(), tol);
    }

protected:
    void assert_layout_() const {
        if(!has_layout()) {
            throw std::runtime_error(
              "Buffer has no layout. Was it default initialized?");
        }
    }

private:
    template<typename OtherDerived>
    friend class BufferBaseCommon;

    /// Splits the last element of @p values off as the new value and
    /// forwards the rest as the offsets for set_element(index_vector,
    /// element_type). Used to implement the variadic overload of
    /// set_element, since a function parameter pack can not be followed by
    /// a deduced fixed parameter.
    template<std::size_t... Is, typename Tuple>
    void set_element_unpack_(std::index_sequence<Is...>, Tuple&& values) {
        set_element(
          index_vector{static_cast<size_type>(std::get<Is>(values))...},
          static_cast<element_type>(std::get<sizeof...(Is)>(values)));
    }

    Derived& derived_() noexcept { return static_cast<Derived&>(*this); }

    /// Access derived for CRTP
    const Derived& derived_() const noexcept {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace tensorwrapper::buffer
