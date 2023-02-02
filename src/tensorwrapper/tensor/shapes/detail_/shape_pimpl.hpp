/*
 * Copyright 2022 NWChemEx-Project
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
#include "tensorwrapper/detail_/hashing.hpp"
#include "tensorwrapper/tensor/allocator/allocator_class.hpp"
#include "tensorwrapper/tensor/shapes/shape.hpp"
#include <TiledArray/sparse_shape.h>

namespace tensorwrapper::tensor::detail_ {

/** @brief Base class for PIMPLs in the Shape hierachy.
 *
 *  The base PIMPL is suitable for tensors which are either dense or whose
 *  sparsity is determined upon filling in the tensor (assuming the tensor
 *  backend supports such a use case).
 *
 *  All PIMPLs in the Shape hierarchy are expected to inherit from this class.
 *  This instance defines the API that the Shape class will use to interact with
 *  the PIMPL. Derived instances of Shape will typically have a derived PIMPL
 *  instance associated with them. The derived PIMPL instances may define
 *  additional functions/state and the derived Shape instance may downcast to
 *  interact with that additional functions/state.
 *
 *  Derived classes should override the following functions as appropriate:
 *
 *  - clone_
 *  - hash_
 *
 *  @tparam FieldType The type of the elements in the tensor. Assumed to be
 *                    either field::Scalar or field::Tensor.
 */
template<typename FieldType>
class ShapePIMPL {
private:
    /// Type of the Shape class being implemented
    using parent_type = Shape<FieldType>;

    /// Type of this PIMPL
    using my_type = ShapePIMPL<FieldType>;

public:
    /// Type used to specify the lengths of each (outer) mode
    using extents_type = typename parent_type::extents_type;

    /// Type used to specify the lengths of each inner mode
    using inner_extents_type = typename parent_type::inner_extents_type;

    /// Type used to return the rank of the Shape
    using size_type = typename parent_type::size_type;

    /// Type used to specify the tiling of the outer modes
    /// TODO: move to Shape
    using tiling_type = std::vector<std::vector<size_type>>;

    /// Type of a pointer to the base of the ShapePIMPL hierarchy
    using pimpl_pointer = typename parent_type::pimpl_pointer;

    /// Type TA uses for specifying the tile sparsity of a tensor
    using ta_shape = TA::SparseShape<float>;

    /// Type used to request slices of the Shape
    using index_type = typename parent_type::index_type;

public:
    /** @brief Creates a new ShapePIMPL with the provided extents.
     *
     *
     *  @param[in] x The extents of each mode of the tensor. When the field is
     *               scalar @p x should specify the extents of each mode. When
     *               the field is tensor @p x should only specify the extents of
     *               the independent modes.
     *  @param[in] y The extents of each inner mode of the tensor. When the
     *               field is scalar @p y will be set to 1 regardless of input.
     *               When the field is tensor @p y should be a map of outer
     *               indices to the shape of the tensor stored in a given index.
     *
     *  @throw None No throw guarantee.
     */
    explicit ShapePIMPL(extents_type x = {}, inner_extents_type y = {}) :
      m_extents_(std::move(x)), m_inner_extents_(std::move(y)) {
        if constexpr(field::is_scalar_field_v<FieldType>)
            m_inner_extents_ = 1;
        else if constexpr(field::is_tensor_field_v<FieldType>) {
            if(m_extents_.size() and !m_inner_extents_.size())
                throw std::runtime_error("ToT Must Have Inner Dimension");
        }

        // Default tiling to span the whole extent of the mode
        m_tiling_ = tiling_type(m_extents_.size());
        for(auto i = 0ul; i < m_extents_.size(); ++i) {
            m_tiling_[i] = {0, m_extents_[i]};
        }
    }

    /** @brief Creates a new ShapePIMPL with the provided tiling and inner
     * extents.
     *
     *
     *  @param[in] x The tiling of each mode of the tensor. When the field is
     *               scalar @p x should specify the tiling of each mode. When
     *               the field is tensor @p x should only specify the tiling of
     *               the independent modes.
     *  @param[in] y The extents of each inner mode of the tensor. When the
     *               field is scalar @p y will be set to 1 regardless of input.
     *               When the field is tensor @p y should be a map of outer
     *               indices to the shape of the tensor stored in a given index.
     *
     *  @throw None No throw guarantee.
     */
    explicit ShapePIMPL(tiling_type x, inner_extents_type y = {}) :
      m_tiling_(std::move(x)), m_inner_extents_(std::move(y)) {
        if constexpr(field::is_scalar_field_v<FieldType>)
            m_inner_extents_ = 1;
        else if constexpr(field::is_tensor_field_v<FieldType>) {
            if(m_tiling_.size() and !m_inner_extents_.size())
                throw std::runtime_error("ToT Must Have Inner Dimension");
        }

        // Set extents to last bound of each tiling
        m_extents_ = extents_type(m_tiling_.size());
        for(auto i = 0ul; i < m_tiling_.size(); ++i) {
            m_extents_[i] = m_tiling_[i].back();
        }
    }

    /** @brief Makes a non-polymorphic deep copy of this instance.
     *
     *  This ctor will deep copy the state in the provided ShapePIMPL
     * instance. If @p other is being passed polymorphically by the base
     * class, then the resulting ShapePIMPL will be the ShapePIMPL slice of @p
     * other.
     *
     *  This ctor is public so that the Shape class can call it for its
     *  non-polymorphic copy.
     *
     *  @param[in] other The PIMPL instance being copied.
     *
     *  @throw std::bad_alloc if there is a problem copying the state. Strong
     *                        throw guarantee.
     */
    ShapePIMPL(const ShapePIMPL& other) = default;

    /// Deleted to avoid slicing
    ///@{
    ShapePIMPL(ShapePIMPL&& other)               = delete;
    ShapePIMPL& operator=(const ShapePIMPL& rhs) = delete;
    ShapePIMPL& operator=(ShapePIMPL&& rhs)      = delete;
    ///@}

    /// Default dtor
    virtual ~ShapePIMPL() noexcept = default;

    /** @brief Polymorphic deep copy.
     *
     *  This function makes a polymorphic deep copy of the current instance.
     *
     *  @return A deep copy of the current instance via a pointer to its
     *          ShapePIMPL base class.
     *
     *  @throw std::bad_alloc if there's a problem allocating memory. Strong
     *                        throw guarantee.
     */
    pimpl_pointer clone() const { return clone_(); }

    /** @brief Returns the lengths of each mode of the tensor.
     *
     *  The extents of a tensor are the lengths of each mode. This function
     *  returns the extents of all modes, when the field is scalar, and the
     *  extents of the independent modes, when the field is tensor.
     *
     *  @return A read-only reference to the tensor's extents.
     *
     *  @throw None No throw gurantee.
     */
    const extents_type& extents() const { return m_extents_; }

    /** @brief Returns the lengths of each inner mode of the tensor.
     *
     *  The inner extents of a tensor are the lengths of each inner mode. This
     *  function returns 1, when the field is scalar, and the extents of the
     *  dependent modes, when the field is tensor.
     *
     *  @return A read-only reference to the tensor's inner extents.
     *
     *  @throw None No throw gurantee.
     */
    const inner_extents_type& inner_extents() const { return m_inner_extents_; }

    /** @brief Returns the tiling of each mode of the tensor.
     *
     *  This function returns the tilings of all modes, when the field is
     *  scalar, and the tilings of the independent modes, when the field is
     *  tensor.
     *
     *  @return A read-only reference to the tensor's extents.
     *
     *  @throw None No throw gurantee.
     */
    const tiling_type& tiling() const { return m_tiling_; }

    size_type field_rank() const {
        if constexpr(field::is_tensor_field_v<FieldType>)
            return m_inner_extents_.size();
        else
            return 0;
    }

    pimpl_pointer slice(const index_type& lo, const index_type& hi) const {
        return slice_(lo, hi);
    }

    /** @brief Non-polymorphic comparison.
     *
     *  This operator is used to compare a ShapePIMPL instance to another
     *  ShapePIMPL instance. The comparision only checks if the extents are the
     *  same in each instance, *i.e.*, if either instance is polymorphic, the
     *  comparison does not consider state outside the ShapePIMPL part of the
     *  object.
     *
     *  @param[in] rhs The instance to compare to.
     *
     *  @return True if @p rhs and the current instance have the same extents,
     *          false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const ShapePIMPL& rhs) const noexcept;

    /** @brief Polymorphic hash operation.
     *
     *  This function will hash the entire state of the current instance,
     *  including any state in the derived classes. For this to work, the
     *  derived class must override `hash_`.
     *
     *  @param[in,out] h The hasher instance to use to hash this instance. After
     *                   this call the interal state of @p h will be updated
     *                   with a hash of the current instance.
     */
    void hash(tensorwrapper::detail_::Hasher& h) const { hash_(h); }

protected:
    /// To be overridden by the derived class to implement hash()
    virtual void hash_(tensorwrapper::detail_::Hasher& h) const {
        h(m_extents_, m_inner_extents_, m_tiling_);
    }

    virtual pimpl_pointer slice_(const index_type&, const index_type&) const;

private:
    /// To be overridden by the derived class to implement clone()
    virtual pimpl_pointer clone_() const;

    /// The extents of the corresponding tensor
    extents_type m_extents_;
    inner_extents_type m_inner_extents_;

    /// The tiling of the corresponding tensor
    tiling_type m_tiling_;
};

#define SHAPE_PIMPL ShapePIMPL<FieldType>

template<typename FieldType>
typename SHAPE_PIMPL::pimpl_pointer SHAPE_PIMPL::clone_() const {
    return pimpl_pointer(new my_type(*this));
}

template<typename FieldType>
typename SHAPE_PIMPL::pimpl_pointer SHAPE_PIMPL::slice_(
  const index_type& _lo, const index_type& _hi) const {
    if(_lo.size() != m_extents_.size())
        throw std::runtime_error("Lo bounds do not match extents");
    if(_hi.size() != m_extents_.size())
        throw std::runtime_error("Hi bounds do not match extents");

    tiling_type new_tiling(m_extents_.size());
    for(auto i = 0ul; i < m_extents_.size(); ++i) {
        if(_lo[i] < 0 or _lo[i] >= m_extents_[i])
            throw std::runtime_error("Invalid lo bound");
        if(_hi[i] > m_extents_[i]) throw std::runtime_error("Invalid hi bound");
        if(_lo[i] > _hi[i])
            throw std::runtime_error("Lo must be smaller than Hi");

        // Shift tiling bounds so the lower bound of the slice is zero.
        // Push back shifted bounds that are greater than zero,
        // until we get to one that is greater than or equal to the new
        // extent of this mode. Then tack the new extent on the end.
        new_tiling[i] = {0};
        for(auto bound : m_tiling_[i]) {
            if(bound < _lo[i]) continue;
            auto shifted_bound = bound - _lo[i];
            if(shifted_bound >= (_hi[i] - _lo[i])) break;
            if(shifted_bound > 0) new_tiling[i].push_back(shifted_bound);
        }
        new_tiling[i].push_back((_hi[i] - _lo[i]));
    }

    inner_extents_type new_inner_extents;
    if constexpr(field::is_scalar_field_v<FieldType>) new_inner_extents = 1;
    if constexpr(field::is_tensor_field_v<FieldType>) {
        /// Make an index_type for the highest included index.
        std::vector<size_type> extents_minus_one(new_tiling.size());
        for(auto i = 0ul; i < new_tiling.size(); ++i) {
            extents_minus_one[i] = new_tiling[i].back() - 1;
        }
        index_type highest_included_index(extents_minus_one);

        /// For each index within the slice, determine their indices with the
        /// lower bound as the new zero and store in a new map with the shape
        /// of the inner tensor.
        for(const auto& [idx, shape] : m_inner_extents_) {
            if(idx < _lo or idx > highest_included_index) continue;
            std::vector<size_type> new_idx(_lo.size());
            for(auto i = 0ul; i < _lo.size(); ++i) {
                new_idx[i] = idx[i] - _lo[i];
            }
            new_inner_extents[index_type(new_idx)] = shape;
        }
    }

    return pimpl_pointer(new my_type(new_tiling, new_inner_extents));
}

template<typename FieldType>
bool SHAPE_PIMPL::operator==(const ShapePIMPL& rhs) const noexcept {
    return m_extents_ == rhs.m_extents_ and
           m_inner_extents_ == rhs.m_inner_extents_ and
           m_tiling_ == rhs.m_tiling_;
}

#undef SHAPE_PIMPL

extern template class ShapePIMPL<field::Scalar>;
extern template class ShapePIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::detail_
