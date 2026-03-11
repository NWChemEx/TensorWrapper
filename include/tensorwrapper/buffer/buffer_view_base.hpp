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
#include <memory>
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/buffer/buffer_base_common.hpp>
#include <tensorwrapper/buffer/detail_/buffer_view_base_pimpl.hpp>
#include <type_traits>

namespace tensorwrapper::buffer {

/** @brief View of a BufferBase that aliases existing state instead of owning
 *         it.
 *
 *  BufferViewBase has the same layout/equality API as BufferBase (has_layout(),
 *  layout(), rank(), operator==, operator!=, approximately_equal) but uses a
 *  PIMPL. The view delegates layout operations to the PIMPL.
 *
 *  BufferViewBase is templated on the type of the aliased buffer (BufferBase or
 *  const BufferBase) for API compatibility; construction from a buffer copies
 *  a non-owning pointer to that buffer's layout into the PIMPL.
 *
 *  The referenced layout (and its owner) must outlive this view. Default-
 *  constructed or moved-from views have no layout (has_layout() is false,
 *  layout() throws).
 *
 *  @tparam BufferBaseType Either BufferBase or const BufferBase.
 */
template<typename BufferBaseType>
class BufferViewBase : public BufferBaseCommon<BufferViewBase<BufferBaseType>> {
private:
    static_assert(std::is_same_v<BufferBaseType, BufferBase> ||
                    std::is_same_v<BufferBaseType, const BufferBase>,
                  "BufferViewBase BufferBaseType must be BufferBase or "
                  "const BufferBase");

    /// Type *this derives from
    using my_base_type = BufferBaseCommon<BufferViewBase<BufferBaseType>>;

    /// Type of the PIMPL
    using pimpl_type            = detail_::BufferViewBasePIMPL<BufferBaseType>;
    using pimpl_reference       = pimpl_type&;
    using const_pimpl_reference = const pimpl_type&;

public:
    using typename my_base_type::const_layout_reference;
    using typename my_base_type::layout_pointer;
    using typename my_base_type::layout_reference;
    using typename my_base_type::layout_type;
    // -------------------------------------------------------------------------
    // -- Ctors and assignment
    // -------------------------------------------------------------------------

    /** @brief Creates a view with no layout.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase() noexcept : m_pimpl_(nullptr) {}

    /** @brief Creates a view that aliases the layout of @p buffer.
     *
     *  @param[in] buffer The buffer whose layout to alias. The layout must
     *                   outlive *this.
     *
     *  @throw None No throw guarantee.
     */
    explicit BufferViewBase(BufferBaseType& buffer) noexcept :
      m_pimpl_(buffer.has_layout() ?
                 std::make_unique<pimpl_type>(&buffer.layout()) :
                 nullptr) {}

    /** Creates a read-only view from a mutable buffer. */
    template<typename OtherBufferBaseType>
        requires(!std::is_const_v<OtherBufferBaseType> &&
                 std::is_const_v<BufferBaseType>)
    explicit BufferViewBase(OtherBufferBaseType& other) noexcept :
      m_pimpl_(other.has_layout() ?
                 std::make_unique<pimpl_type>(&other.layout()) :
                 nullptr) {}

    explicit BufferViewBase(layout_pointer layout) noexcept :
      m_pimpl_(std::make_unique<pimpl_type>(layout)) {}

    /** @brief Creates a view that aliases the same layout as @p other.
     *
     *  @param[in] other The view to copy.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase(const BufferViewBase& other) noexcept :
      m_pimpl_(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

    /** @brief Creates a view by taking the PIMPL from @p other.
     *
     *  After construction *this aliases the layout @p other did, and @p other
     *  has no layout.
     *
     *  @param[in,out] other The view to move from.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase(BufferViewBase&& other) noexcept = default;

    /** @brief Makes *this alias the same layout as @p rhs.
     *
     *  @param[in] rhs The view to copy.
     *
     *  @return *this.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase& operator=(const BufferViewBase& rhs) noexcept {
        if(this != &rhs) {
            m_pimpl_ = rhs.m_pimpl_ ? rhs.m_pimpl_->clone() : nullptr;
        }
        return *this;
    }

    /** @brief Replaces the PIMPL in *this with that of @p rhs.
     *
     *  @param[in,out] rhs The view to move from.
     *
     *  @return *this.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase& operator=(BufferViewBase&& rhs) noexcept = default;

    /** @brief Is *this different from @p rhs?
     *
     *  @param[in] rhs The view to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const BufferViewBase& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    friend my_base_type;
    friend class BufferBase;

    // -------------------------------------------------------------------------
    // -- BufferBaseCommon hooks
    // -------------------------------------------------------------------------

    bool has_layout_() const noexcept {
        return m_pimpl_ != nullptr && m_pimpl_->has_layout();
    }

    layout_reference layout_() { return pimpl_().layout(); }

    const_layout_reference layout_() const { return pimpl_().layout(); }

    // Will be polymorphic eventually
    template<typename OtherBufferBaseType>
    bool approximately_equal_(const BufferViewBase<OtherBufferBaseType>& rhs,
                              double) const {
        return *this == rhs;
    }

    // Will be polymorphic eventually
    bool approximately_equal_(const BufferBase& rhs, double) const {
        return *this == rhs;
    }

private:
    void assert_pimpl_() const {
        if(!m_pimpl_) {
            throw std::runtime_error(
              "BufferViewBase has no PIMPL. Was it default initialized?");
        }
    }
    pimpl_reference pimpl_() {
        assert_pimpl_();
        return *m_pimpl_;
    }

    const_pimpl_reference pimpl_() const {
        assert_pimpl_();
        return *m_pimpl_;
    }

    /// PIMPL holding non-owning pointer to the aliased layout
    std::unique_ptr<pimpl_type> m_pimpl_;
};

// Out-of-line definition so both BufferBase and BufferViewBase are complete.

template<typename BufferBaseType>
bool BufferBase::approximately_equal_(const BufferViewBase<BufferBaseType>& rhs,
                                      double tol) const {
    if(!rhs.has_layout()) return !has_layout();
    return !this->layout().are_different(rhs.layout());
}

} // namespace tensorwrapper::buffer
