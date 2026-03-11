/*
 * Copyright 2026 NWChemEx-Project
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
#include <tensorwrapper/buffer/local.hpp>
#include <tensorwrapper/buffer/replicated_common.hpp>
#include <tensorwrapper/types/preserve_const.hpp>
namespace tensorwrapper::buffer {
namespace detail_ {

template<typename ReplicatedType>
class ReplicatedViewPIMPL;

} // namespace detail_

/** @brief A view of a Replicated buffer.
 *
 *  @tparam ReplicatedType The type of the Replicated buffer to view. Expected
 * to be unqualified Replicated or const Replicated.
 *
 *  This class is a view of a Replicated buffer. It is used to create a view of
 *  a Replicated buffer.
 */
template<typename ReplicatedType>
class ReplicatedView
  : public ReplicatedCommon<ReplicatedView<ReplicatedType>>,
    public LocalView<types::preserve_const_t<ReplicatedType, Local>> {
private:
    using my_type          = ReplicatedView<ReplicatedType>;
    using common_base_type = ReplicatedCommon<my_type>;
    using local_base_type =
      LocalView<types::preserve_const_t<ReplicatedType, Local>>;
    using my_base_type = LocalView<local_base_type>;

public:
    /// Pull in base's types
    ///@{
    using typename common_base_type::const_element_reference;
    using typename common_base_type::element_reference;
    using typename common_base_type::element_type;
    using typename common_base_type::index_vector;
    using typename common_base_type::size_type;
    ///@}

    /// Type of the PIMPL
    using pimpl_type = detail_::ReplicatedViewPIMPL<ReplicatedType>;

    /// Type of a pointer to the PIMPL
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /** @brief Default constructor.
     *
     *  This constructor will create a view with no layout and no elements.
     *
     *  @throw None No throw guarantee.
     */
    ReplicatedView();

    /** @brief Slice construction.
     *
     *  This ctor will create a view of the @p replicated buffer starting at
     *  the @p first_elem and ending at the @p last_elem.
     *
     *  @param[in] replicated The replicated buffer to slice. The replicated
     *                        buffer must outlive the view.
     *  @param[in] first_elem The first element of the slice.
     *  @param[in] last_elem The last element of the slice.
     *
     *  @throw std::runtime_error if the slice is invalid. Strong throw
     * guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the PIMPL. Strong
     * throw guarantee.
     */
    ReplicatedView(ReplicatedType& replicated, index_vector first_elem,
                   index_vector last_elem);

    /** @brief Creates a new view implemented by @p pimpl.
     *
     *  @param[in] pimpl A pointer to the PIMPL to use as the backing store.
     *
     *  @throw None No throw guarantee.
     */
    ReplicatedView(pimpl_pointer pimpl);

    /** @brief Creates a new view by copying the state of @p other.
     *
     *  This ctor will create a new view by copying the state of @p other.
     *  After this operation *this will alias the same object @p other did.
     *
     *  @param[in] other The view to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    ReplicatedView(const ReplicatedView& other);

    /** @brief Overwrites the state of *this with the state of @p rhs.
     *
     *  This operator will overwrite the state of *this by moving the pointers
     *  in @p rhs. After this operation *this will alias
     *  the same object @p rhs did.
     *
     *  @param[in] rhs The view to move from.
     *
     *  @throw None No throw guarantee.
     */
    ReplicatedView(ReplicatedView&& other) noexcept;

    /** @brief Overwrites the state of *this with the state of @p rhs.
     *
     *  This operator will overwrite the state of *this by copying the pointers
     *  in @p rhs. This is a shallow copy. After this operation *this will alias
     *  the same object @p rhs did. It is worth noting the aliased object is
     *  untouched after this operation.
     *
     *  @param[in] rhs The view to copy.
     *
     *  @return *this after making it alias the state in @p rhs.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    ReplicatedView& operator=(const ReplicatedView& rhs);

    /** @brief Overwrites the state of *this with the state of @p rhs.
     *
     *  This operator will overwrite the state of *this with the state of
     *  @p rhs. After this operation *this will alias the same object @p rhs
     * did.
     *
     *  @param[in] rhs The view to copy.
     *
     *  @return *this after making it alias the state in @p rhs.
     *
     *  @throw None No throw guarantee.
     */
    ReplicatedView& operator=(ReplicatedView&& rhs) noexcept;

    /// No-throw dtor.
    ~ReplicatedView() noexcept;

protected:
    friend common_base_type;

    /// Implements get_elem for the view.
    const_element_reference get_elem_(index_vector index) const;

    /// Implements set_elem for the view.
    void set_elem_(index_vector index, element_type value);

private:
    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Throws if *this does not have a PIMPL.
    void assert_pimpl_() const;

    /// The PIMPL holding the data for the view.
    pimpl_pointer m_pimpl_;
};

extern template class ReplicatedView<Replicated>;
extern template class ReplicatedView<const Replicated>;

} // namespace tensorwrapper::buffer
