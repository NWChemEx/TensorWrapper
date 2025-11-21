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
    using traits_type  = types::ClassTraits<MDBuffer>;
    using my_type      = MDBuffer;

public:
    /// Add types to public API
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
    using hash_type   = std::size_t;

    MDBuffer() noexcept;

    template<concepts::FloatingPoint T>
    MDBuffer(std::vector<T> elements, shape_type shape) :
      MDBuffer(buffer_type(std::move(elements)), std::move(shape)) {}

    MDBuffer(buffer_type buffer, shape_type shape);

    MDBuffer(const MDBuffer& other)     = default;
    MDBuffer(MDBuffer&& other) noexcept = default;

    MDBuffer& operator=(const MDBuffer& other)     = default;
    MDBuffer& operator=(MDBuffer&& other) noexcept = default;

    ~MDBuffer() override = default;

    // -------------------------------------------------------------------------
    // -- State Accessors
    // -------------------------------------------------------------------------

    const_shape_view shape() const;

    size_type size() const noexcept;

    const_reference get_elem(index_vector index) const;

    void set_elem(index_vector index, value_type new_value);

    buffer_view get_mutable_data();

    const_buffer_view get_immutable_data() const;

    // -------------------------------------------------------------------------
    // -- Utility Methods
    // -------------------------------------------------------------------------

    bool operator==(const my_type& rhs) const noexcept;

protected:
    buffer_base_pointer clone_() const override;

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

    string_type to_string_() const override;

    std::ostream& add_to_stream_(std::ostream& os) const override;

    // Returns the hash for the current state of *this, computing first if
    // needed.
    hash_type get_hash_() const {
        if(m_recalculate_hash_ or !m_hash_caching_) update_hash_();
        return m_hash_;
    }

private:
    size_type coordinate_to_ordinal_(index_vector index) const;

    // Computes the hash for the current state of *this
    void update_hash_() const;

    // Designates that the state may have changed and to recalculate the hash.
    // This function is really just for readability and clarity.
    void mark_for_rehash_() const { m_recalculate_hash_ = true; }

    // Designates that state changes are not trackable and we should recalculate
    // the hash each time.
    void turn_off_hash_caching_() const { m_hash_caching_ = false; }

    // Tracks whether the hash needs to be redetermined
    mutable bool m_recalculate_hash_ = true;

    // Tracks whether hash caching has been turned off
    mutable bool m_hash_caching_ = true;

    // Holds the computed hash value for this instance's state
    mutable hash_type m_hash_ = 0;

    shape_type m_shape_;

    buffer_type m_buffer_;
};

} // namespace tensorwrapper::buffer
