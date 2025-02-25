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

    /// Returns a mutable pointer to the first element in contiguous memory
    pointer data() noexcept { return data_(); }

    /// Returns a read-only pointer to the first element in contiguous memory
    const_pointer data() const noexcept { return data_(); }

    template<typename... Args>
    reference at(Args&&... args) {
        return get_elem_(
          index_vector{detail_::to_size_t(std::forward<Args>(args))...});
    }

    template<typename... Args>
    const_reference at(Args&&... args) const {
        return get_elem_(
          index_vector{detail_::to_size_t(std::forward<Args>(args))...});
    }

protected:
    /// Derived class can override if it likes
    virtual size_type size_() const noexcept { return layout().shape().size(); }

    /// Derived class should implement according to data() description
    virtual pointer data_() noexcept = 0;

    /// Derived class should implement according to data() const description
    virtual const_pointer data_() const noexcept = 0;

    /// Derived class should implement according to operator()()
    virtual reference get_elem_(index_vector index) = 0;

    /// Derived class should implement according to operator()()const
    virtual const_reference get_elem_(index_vector index) const = 0;
};

} // namespace tensorwrapper::buffer