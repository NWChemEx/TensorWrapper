/*
 * Copyright 2024 NWChemEx Community
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
#include <functional>
#include <numeric>
#include <tensorwrapper/shape/shape_base.hpp>
#include <vector>

namespace tensorwrapper::shape {

/** @brief Describes the shape of a "traditional" tensor.
 *
 *  Tensors are traditionally thought of as being (hyper-)rectangular arrays of
 *  scalars. The geometry of such a shape is described by stating the
 *  geometric dimension of the (hyper-)rectangle and the number of elements in
 *  the array.
 */
class Smooth : public ShapeBase {
public:
    // Pull in base class's types
    using ShapeBase::const_labeled_reference;
    using ShapeBase::dsl_reference;
    using ShapeBase::label_type;
    using ShapeBase::rank_type;
    using ShapeBase::size_type;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /** @brief Constructs *this with a statically specified number of extents.
     *
     *  This ctor is used to create a Smooth object by explicitly providing
     *  the extents. The number of extents must be known at compile time. For
     *  a dynamic number of extents use the range ctor.
     *
     *  @param[in] il The extents of the modes.
     *
     *  @throw std::runtime_error if there is a problem allocating the internal
     *                            state. Strong throw guarantee.
     */
    Smooth(std::initializer_list<size_type> il) :
      Smooth(il.begin(), il.end()) {}

    /** @brief Range ctor.
     *
     *  @tparam BeginItrType Expected to be a forward iterator which can be
     *                       dereferenced to an object of size_type.
     *  @tparam EndItrType Expected to be a type which can be compared to an
     *                     object of type BeginItrType.
     *
     *  This ctor is used to construct a Smooth object with the extent of each
     *  mode provided by a pair of iterators.
     *
     *  @param[in] begin An iterator pointing to the extent of mode 0.
     *  @param[in] end An iterator pointing to just past the extent of
     *                         the last mode.
     *
     *  @throw ??? If iterating, dereferencing the begin iterator, or comparing
     *             the iterators throws. Same throw guarantee as the iterators
     *             involved in the throw.
     *  @throw std::bad_alloc if there is a problem allocating the internal
     *             state. Strong throw guarantee.
     */
    template<typename BeginItrType, typename EndItrType>
    Smooth(BeginItrType&& begin, EndItrType&& end) :
      Smooth(extents_type(std::forward<BeginItrType>(begin),
                          std::forward<EndItrType>(end))) {}

    /// Defaulted no-throw dtor.
    ~Smooth() noexcept = default;

    // -------------------------------------------------------------------------
    // -- Accessor methods
    // -------------------------------------------------------------------------

    /** @brief Returns the extent of the @p i -th mode.
     *
     *  @param[in] i The mode the user wants the extent of. @p i must be in the
     *               range [0, rank()).
     *
     *  @return The extent of the requested mode.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, range()).
     *                           Strong throw guarantee.
     */
    rank_type extent(size_type i) const { return m_extents_.at(i); }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Exchanges the state in *this with that of @p other.
     *
     *  @param[in,out] other The object to take the state from. After this
     *                 method is called @p other will have the same state that
     *                 *this previously had.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Smooth& other) noexcept { m_extents_.swap(other.m_extents_); }

    /** @brief Is *this the same shape as @p rhs?
     *
     *  @note This is a non-polymorphic value comparison, i.e., any state in
     *        *this or @p rhs that resides in derived classes is NOT considered
     *        in this comparison.
     *
     *  Two Smooth objects are value equal if they contain the same number of
     *  modes and if their @f$i@f$-th modes have the same extent for all @f$i@f$
     *  in the range [0, rank()).
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     */
    bool operator==(const Smooth& rhs) const noexcept {
        return m_extents_ == rhs.m_extents_;
    }

    /** @brief Is *this different from @p rhs?
     *
     *  @note This is a non-polymorphic value comparison, i.e., any state in
     *        *this or @p rhs that resides in derived classes is NOT considered
     *        in this comparison.
     *
     *  This method defines "different" as not value equal. See `operator==` for
     *  the definition of value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Smooth& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    /// Implement clone() by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<Smooth>(*this);
    }

    /// Implement rank by counting number of extents held by *this
    rank_type get_rank_() const noexcept override {
        return rank_type(m_extents_.size());
    }

    /// Implement size by taking the product of the extents held by *this
    size_type get_size_() const noexcept override {
        return std::accumulate(m_extents_.begin(), m_extents_.end(),
                               size_type(1), std::multiplies<size_type>());
    }

    smooth_reference as_smooth_() override { return smooth_reference(*this); }

    virtual const_smooth_reference as_smooth_() const override {
        return const_smooth_reference(*this);
    }

    /// Implements are_equal by calling ShapeBase::are_equal_impl_
    bool are_equal_(const ShapeBase& rhs) const noexcept override {
        return are_equal_impl_<Smooth>(rhs);
    }

    /// Implements addition_assignment by considering permutations
    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference rhs) override;

    /// Implements permute_assignment by permuting the extents in @p rhs.
    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

    /// Implements to_string
    string_type to_string_() const override {
        string_type buffer("{");
        for(auto x : m_extents_) buffer += string_type(" ") + std::to_string(x);
        buffer += string_type("}");
        return buffer;
    }

private:
    /// Type used to hold the extents of *this
    using extents_type = std::vector<size_type>;

    /// Constructs *this given an object of extents_type
    explicit Smooth(extents_type extents) : m_extents_(std::move(extents)) {}

    /// The length of each mode
    extents_type m_extents_;
};

} // namespace tensorwrapper::shape
