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
#include <tensorwrapper/detail_/integer_utilities.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::buffer {

/** @brief Denotes that a buffer is held contiguously.
 *
 *  Contiguous buffers are such that given a pointer to the first element `p`,
 *  the `i`-th element (`i` is zero based) is given by dereferencing the
 *  pointer `p + i`. Note that contiguous buffers are always vectors and storing
 *  higher rank tensors in a contiguous buffer requires "vectorization" of the
 *  tensor. In C++ vectorization is usually done in row-major format.
 *
 *  @tparam FloatType the type of elements in the buffer.
 */
template<typename FloatType>
class Contiguous : public Replicated {
private:
    /// Type *this derives from
    using my_base_type = Replicated;

public:
    /// Type of each element
    using element_type = FloatType;

    /// Type of a mutable reference to an object of type element_type
    using reference = element_type&;

    /// Type of a read-only reference to an object of type element_type
    using const_reference = const element_type&;

    using elements_type = std::vector<element_type>;

    /// Type of a pointer to a mutable element_type object
    using pointer = element_type*;

    /// Type of a pointer to a read-only element_type object
    using const_pointer = const element_type*;

    /// Type used for offsets and indexing
    using size_type = std::size_t;

    /// Type of a multi-dimensional index
    using index_vector = std::vector<size_type>;

    // Pull in base's ctors
    using my_base_type::my_base_type;

    /// Returns the number of elements in contiguous memory
    size_type size() const noexcept { return size_(); }

    /** @brief Returns a mutable pointer to the first element in contiguous
     *         memory
     *
     *  @warning Returning a mutable pointer to the underlying data makes it
     *           no longer possible for *this to reliably track changes to that
     *           data. Calling this method may have performance implications, so
     *           use only when strictly required.
     *
     *  @return A read/write pointer to the data.
     *
     *  @throw None No throw guarantee.
     */
    pointer get_mutable_data() noexcept { return get_mutable_data_(); }

    /** @brief Returns an immutable pointer to the first element in contiguous
     *         memory
     *
     *  @return A read-only pointer to the data.
     *
     *  @throw None No throw guarantee.
     */
    const_pointer get_immutable_data() const noexcept {
        return get_immutable_data_();
    }

    /** @brief Retrieves a tensor element by offset.
     *
     *  This method is used to access the element in an immutable way.
     *
     *  @param[in] index The offset of the element being retrieved.
     *
     *  @return A read-only reference to the element.
     *
     *  @throw std::runtime_error if the number of indices does not match the
     *                            rank of the tensor. Strong throw guarantee.
     */
    const_reference get_elem(index_vector index) const {
        if(index.size() != this->rank())
            throw std::runtime_error("Number of offsets must match rank");
        return get_elem_(index);
    }

    /** @brief Sets a tensor element by offset.
     *
     *  This method is used to change the value of an element.
     *
     *  @param[in] index The offset of the element being updated.
     *  @param[in] new_value The new value of the element.
     *
     *  @throw std::runtime_error if the number of indices does not match the
     *                            rank of the tensor. Strong throw guarantee.
     */
    void set_elem(index_vector index, element_type new_value) {
        if(index.size() != this->rank())
            throw std::runtime_error("Number of offsets must match rank");
        return set_elem_(index, new_value);
    }

    /** @brief Retrieves a tensor element by ordinal offset.
     *
     *  This method is used to access the element in an immutable way.
     *
     *  @param[in] index The ordinal offset of the element being retrieved.
     *
     *  @return A read-only reference to the element.
     *
     *  @throw std::runtime_error if the index is greater than the number of
     *                            elements. Strong throw guarantee.
     */
    const_reference get_data(size_type index) const {
        if(index >= this->size())
            throw std::runtime_error("Index greater than number of elements");
        return get_data_(std::move(index));
    }

    /** @brief Sets a tensor element by ordinal offset.
     *
     *  This method is used to change the value of an element.
     *
     *  @param[in] index The ordinal offset of the element being updated.
     *  @param[in] new_value The new value of the element.
     *
     *  @throw std::runtime_error if the index is greater than the number of
     *                            elements. Strong throw guarantee.
     */
    void set_data(size_type index, element_type new_value) {
        if(index >= this->size())
            throw std::runtime_error("Index greater than number of elements");
        set_data_(index, new_value);
    }

    /** @brief Sets all elements to a value.
     *
     *  @param[in] value The new value of all elements.
     *
     *  @throw None No throw guarantee.
     */
    void fill(element_type value) { fill_(value); }

    void copy(elements_type& values) { copy_(values); }

protected:
    /// Derived class can override if it likes
    virtual size_type size_() const noexcept { return layout().shape().size(); }

    /// Derived class should implement according to data() description
    virtual pointer get_mutable_data_() noexcept = 0;

    /// Derived class should implement according to data() const description
    virtual const_pointer get_immutable_data_() const noexcept = 0;

    /// Derived class should implement according to get_elem()
    virtual const_reference get_elem_(index_vector index) const = 0;

    /// Derived class should implement according to set_elem()
    virtual void set_elem_(index_vector index, element_type new_value) = 0;

    /// Derived class should implement according to get_data()
    virtual const_reference get_data_(size_type index) const = 0;

    /// Derived class should implement according to set_data()
    virtual void set_data_(size_type index, element_type new_value) = 0;

    /// Derived class should implement according to fill()
    virtual void fill_(element_type) = 0;

    virtual void copy_(elements_type& values) = 0;
};

#define DECLARE_CONTIG_BUFFER(TYPE) extern template class Contiguous<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_CONTIG_BUFFER);

#undef DECLARE_CONTIG_BUFFER

} // namespace tensorwrapper::buffer