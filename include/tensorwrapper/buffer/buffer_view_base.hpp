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
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/buffer/buffer_base_common.hpp>
#include <type_traits>

namespace tensorwrapper::buffer {

/** @brief View of a BufferBase that aliases existing state instead of owning
 * it.
 *
 *  BufferViewBase has the same layout/equality API as BufferBase (has_layout(),
 *  layout(), rank(), operator==, operator!=, approximately_equal) but holds a
 *  non-owning pointer to a BufferBase and delegates all operations to it.
 *
 *  BufferViewBase is templated on the type of the aliased buffer, which must
 *  be either BufferBase or const BufferBase. This controls whether the view is
 *  a mutable or const view of the underlying BufferBase.
 *
 *  The aliased buffer must outlive this view. Default-constructed or
 *  moved-from views have no aliased buffer (has_layout() is false, layout()
 *  throws).
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
    using typename my_base_type::const_layout_reference;

    using aliased_type    = BufferBaseType;
    using aliased_pointer = aliased_type*;

public:
    // -------------------------------------------------------------------------
    // -- Ctors and assignment
    // -------------------------------------------------------------------------

    /** @brief Creates a view that aliases no buffer.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase() noexcept : m_aliased_(nullptr) {}

    /** @brief Creates a view that aliases @p buffer.
     *
     *  @param[in] buffer The buffer to alias. Must outlive *this.
     *
     *  @throw None No throw guarantee.
     */
    explicit BufferViewBase(aliased_type& buffer) noexcept :
      m_aliased_(&buffer) {}

    /** @brief Creates a view that aliases the same buffer as @p other.
     *
     *  @param[in] other The view to copy.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase(const BufferViewBase& other) noexcept = default;

    /** @brief Creates a view by taking the alias from @p other.
     *
     *  After construction *this aliases the buffer @p other did, and @p other
     *  aliases no buffer.
     *
     *  @param[in,out] other The view to move from.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase(BufferViewBase&& other) noexcept = default;

    /** @brief Makes *this alias the same buffer as @p rhs.
     *
     *  @param[in] rhs The view to copy.
     *
     *  @return *this.
     *
     *  @throw None No throw guarantee.
     */
    BufferViewBase& operator=(const BufferViewBase& rhs) noexcept = default;

    /** @brief Replaces the alias in *this with that of @p rhs.
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
        return m_aliased_ != nullptr && m_aliased_->has_layout();
    }

    const_layout_reference layout_() const {
        if(m_aliased_ == nullptr) {
            throw std::runtime_error(
              "Buffer has no layout. Was it default initialized?");
        }
        return m_aliased_->layout();
    }

    template<typename OtherBufferBase>
    bool approximately_equal_(const BufferViewBase<OtherBufferBase>& rhs,
                              double tol) const {
        if(m_aliased_ == nullptr) return !rhs.has_layout();
        return m_aliased_->approximately_equal(*rhs.m_aliased_, tol);
    }

    bool approximately_equal_(const BufferBase& rhs, double tol) const {
        if(m_aliased_ == nullptr) return !rhs.has_layout();
        return m_aliased_->approximately_equal(rhs, tol);
    }

private:
    /// The buffer *this aliases (non-owning)
    aliased_pointer m_aliased_;
};

// Out-of-line definition so both BufferBase and BufferViewBase are complete
template<typename BufferBaseType>
bool BufferBase::approximately_equal_(const BufferViewBase<BufferBaseType>& rhs,
                                      double tol) const {
    if(!rhs.has_layout()) return !has_layout();
    return approximately_equal_(
      *static_cast<const BufferBaseType*>(rhs.m_aliased_), tol);
}

} // namespace tensorwrapper::buffer
