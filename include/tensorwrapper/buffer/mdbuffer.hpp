/*
 * Copyright 2025 NWChemEx-Project
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
#include <tensorwrapper/buffer/replicated.hpp>
#include <tensorwrapper/concepts/floating_point.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/types/mdbuffer_traits.hpp>

namespace tensorwrapper::buffer {

/** @brief A multidimensional (MD) buffer.
 *
 *  This class is a dense multidimensional buffer of floating-point values.
 */
class MDBuffer : public Replicated {
private:
    /// Type *this derives from
    using my_base_type = Replicated;

    /// Type defining the types for the public API of *this
    using traits_type = types::ClassTraits<MDBuffer>;

    /// Type of *this
    using my_type = MDBuffer;

public:
    /// Add types from traits_type to public API
    ///@{
    using value_type        = typename traits_type::value_type;
    using reference         = typename traits_type::reference;
    using const_reference   = typename traits_type::const_reference;
    using buffer_type       = typename traits_type::buffer_type;
    using buffer_view       = typename traits_type::buffer_view;
    using const_buffer_view = typename traits_type::const_buffer_view;
    using pimpl_type        = typename traits_type::pimpl_type;
    using pimpl_pointer     = typename traits_type::pimpl_pointer;
    using rank_type         = typename traits_type::rank_type;
    using shape_type        = typename traits_type::shape_type;
    using const_shape_view  = typename traits_type::const_shape_view;
    using size_type         = typename traits_type::size_type;
    ///@}

    using index_vector = std::vector<size_type>;
    using typename my_base_type::label_type;
    using string_type = std::string;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /** @brief Creates an empty multi-dimensional buffer.
     *
     *  The resulting buffer will have a shape of rank 0, but a size of 0. Thus
     *  the buffer can NOT be used to store any elements (including treating
     *  *this as a scalar). The resulting buffer can be assigned to or moved
     *  to to populate it.
     *
     *  @throw None No throw guarantee.
     */
    MDBuffer() noexcept;

    /** @brief Treats allocated memory like a multi-dimensional buffer.
     *
     *  @tparam T The type of the elements in the buffer. Must satisfy the
     *            FloatingPoint concept.
     *
     *  This ctor will use @p element to create a buffer_type object and then
     *  pass that along with @p shape to the main ctor.
     *
     *  @param[in] elements The elements to be used as the backing store.
     *  @param[in] shape The shape of *this.
     *
     *  @throw std::invalid_argument if the size of @p elements does not match
     *                               the size implied by @p shape. Strong throw
     *                               guarantee.
     *  @throw std::bad_alloc if there is a problem allocating memory for the
     *                        internal state. Strong throw guarantee.
     */
    template<concepts::FloatingPoint T>
    MDBuffer(std::vector<T> elements, shape_type shape) :
      MDBuffer(buffer_type(std::move(elements)), std::move(shape)) {}

    /** @brief The main ctor.
     *
     *  This ctor will create *this using @p buffer as the backing store and
     *  @p shape to describe the geometry of the multidimensional array.
     *
     *  All other ctors (aside from copy and move) delegate to this one.
     *
     *  @param[in] buffer The buffer to be used as the backing store.
     *  @param[in] shape The shape of *this.
     *
     *  @throw std::invalid_argument if the size of @p buffer does not match
     *                               the size implied by @p shape. Strong throw
     *                               guarantee.
     *  @throw std::bad_alloc if there is a problem allocating memory for the
     *                        internal state. Strong throw guarantee.
     */
    MDBuffer(buffer_type buffer, shape_type shape);

    /** @brief Initializes *this to a deep copy of @p other.
     *
     *  This ctor will initialize *this to be a deep copy of @p other.
     *
     *  @param[in] other The MDBuffer to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating memory for the
     *                        internal state. Strong throw guarantee.
     */
    MDBuffer(const MDBuffer& other) = default;

    /** @brief Move ctor.
     *
     *  This ctor will initialize *this by taking the state from @p other.
     *  After this ctor is called @p other is left in a valid but unspecified
     *  state.
     *
     *  @param[in,out] other The MDBuffer to move from.
     *
     *  @throw None No throw guarantee.
     */
    MDBuffer(MDBuffer&& other) noexcept = default;

    /** @brief Copy assignment.
     *
     *  This operator will make *this a deep copy of @p other.
     *
     *  @param[in] other The MDBuffer to copy.
     *
     *  @return *this after the assignment.
     *
     *  @throw std::bad_alloc if there is a problem allocating memory for the
     *                        internal state. Strong throw guarantee.
     */
    MDBuffer& operator=(const MDBuffer& other) = default;

    /** @brief Move assignment.
     *
     *  This operator will make *this take the state from @p other. After
     *  this operator is called @p other is left in a valid but unspecified
     *  state.
     *
     *  @param[in,out] other The MDBuffer to move from.
     *
     *  @return *this after the assignment.
     *
     *  @throw None No throw guarantee.
     */
    MDBuffer& operator=(MDBuffer&& other) noexcept = default;

    /** @brief Defaulted dtor.
     *
     *  @throw None No throw guarantee.
     */
    ~MDBuffer() override = default;

    // -------------------------------------------------------------------------
    // -- State Accessors
    // -------------------------------------------------------------------------

    /** @brief Returns (a view of) the shape of *this.
     *
     *  The shape of *this describes the geometry of the underlying
     *  multidimensional array.
     *
     *  @return A view of the shape of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating memory for the
     *                        returned view. Strong throw guarantee.
     */
    const_shape_view shape() const;

    /** @brief The total number of elements in *this.
     *
     *  The total number of elements is the product of the extents of each
     *  mode of *this.
     *
     *  @return The total number of elements in *this.
     *
     *  @throw None No throw guarantee.
     */
    size_type size() const noexcept;

    /** @brief Returns the element with the offsets specified by @p index.
     *
     *  This method will retrieve a const reference to the element at the
     *  offsets specified by @p index. The length of @p index must be equal
     *  to the rank of *this and each entry in @p index must be less than the
     *  extent of the corresponding mode of *this.
     *
     *  This method can only be used to retrieve elements from *this. To modify
     *  elements use set_elem().
     *
     *  @param[in] index The offsets into each mode of *this for the desired
     *                   element.
     *
     *  @return A const reference to the element at the specified offsets.
     */
    const_reference get_elem(index_vector index) const;

    /** @brief Sets the specified element to @p new_value.
     *
     *  This method will set the element at the offsets specified by @p index.
     *  The length of @p index must be equal to the rank of *this and each
     *  entry in @p index must be less than the extent of the corresponding
     *  mode of *this.
     *
     *  @param[in] index The offsets into each mode of *this for the desired
     *                   element.
     *  @param[in] new_value The new value for the specified element.
     *
     *  @throw std::out_of_range if any entry in @p index is invalid. Strong
     *                           throw guarantee.
     */
    void set_elem(index_vector index, value_type new_value);

    /** @brief Returns a view of the data.
     *
     *  This method is deprecated. Use set_slice instead.
     */
    [[deprecated]] buffer_view get_mutable_data();

    /** @brief Returns a read-only view of the data.
     *
     *  This method is deprecated. Use get_slice instead.
     */
    [[deprecated]] const_buffer_view get_immutable_data() const;

    // -------------------------------------------------------------------------
    // -- Utility Methods
    // -------------------------------------------------------------------------

    /** @brief Compares two MDBuffer objects for exact equality.
     *
     *  Two MDBuffer objects are exactly equal if they have the same shape and
     *  if all of their corresponding elements are bitwise identical.
     *  In practice, the implementation stores a hash of the elements in the
     *  tensor and compares the hashes for equality rather than checking each
     *  element individually.
     *
     *  @param[in] rhs The MDBuffer to compare against.
     *
     *  @return True if *this and @p rhs are exactly equal and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const my_type& rhs) const noexcept;

protected:
    /// Makes a deep polymorphic copy of *this
    buffer_base_pointer clone_() const override;

    /// Implements are_equal by checking that rhs is an MDBuffer and then
    /// calling operator==
    bool are_equal_(const_buffer_base_reference rhs) const noexcept override;

    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;
    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;
    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

    dsl_reference scalar_multiplication_(label_type this_labels, double scalar,
                                         const_labeled_reference rhs) override;

    /// Calls add_to_stream_ on a stringstream to implement
    string_type to_string_() const override;

    /// Uses Eigen's printing capabilities to add to stream
    std::ostream& add_to_stream_(std::ostream& os) const override;

private:
    /// Type for storing the hash of *this
    using hash_type = std::size_t;

    /// Logic for validating that an index is within the bounds of the shape
    void check_index_(const index_vector& index) const;

    /// Converts a coordinate index to a linear (ordinal) index
    size_type coordinate_to_ordinal_(index_vector index) const;

    /// Returns the hash for the current state of *this, computing first if
    /// needed.
    hash_type get_hash_() const {
        if(m_recalculate_hash_ or !m_hash_caching_) update_hash_();
        return m_hash_;
    }

    /// Computes the hash for the current state of *this
    void update_hash_() const;

    /// Designates that the state may have changed and to recalculate the hash.
    /// This function is really just for readability and clarity.
    void mark_for_rehash_() const { m_recalculate_hash_ = true; }

    /// Designates that state changes are not trackable and we should
    /// recalculate the hash each time.
    void turn_off_hash_caching_() const { m_hash_caching_ = false; }

    /// Tracks whether the hash needs to be redetermined
    mutable bool m_recalculate_hash_ = true;

    /// Tracks whether hash caching has been turned off
    mutable bool m_hash_caching_ = true;

    /// Holds the computed hash value for this instance's state
    mutable hash_type m_hash_ = 0;

    /// How the hyper-rectangular array is shaped
    shape_type m_shape_;

    /// The flat buffer holding the elements of *this
    buffer_type m_buffer_;
};

} // namespace tensorwrapper::buffer
