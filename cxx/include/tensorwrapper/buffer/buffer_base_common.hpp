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

    Derived& derived_() noexcept { return static_cast<Derived&>(*this); }

    /// Access derived for CRTP
    const Derived& derived_() const noexcept {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace tensorwrapper::buffer
