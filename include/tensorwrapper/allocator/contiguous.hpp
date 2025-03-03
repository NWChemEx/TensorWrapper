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
#include <tensorwrapper/allocator/replicated.hpp>
#include <tensorwrapper/types/il_traits.hpp>

namespace tensorwrapper::allocator {

/** @brief Allocator that can create Contiguous buffers.
 *
 *  @tparam FloatType Type of the elements in the contiguous buffer.
 */
template<typename FloatType>
class Contiguous : public Replicated {
private:
    /// Type of *this
    using my_type = Contiguous<FloatType>;

    /// Type *this derives from
    using base_type = Replicated;

public:
    /// Pull in base types
    ///@{
    using base_type::buffer_base_pointer;
    using base_type::const_layout_reference;
    using base_type::layout_pointer;
    ///@}

    /// Type of each element in the tensor
    using element_type = FloatType;

    /// Type of the buffer associated with *this
    using contiguous_buffer_type = buffer::Contiguous<element_type>;
    using contiguous_pointer     = std::unique_ptr<contiguous_buffer_type>;

    /// Type of initializer lists
    using rank0_il = typename types::ILTraits<element_type, 0>::type;
    using rank1_il = typename types::ILTraits<element_type, 1>::type;
    using rank2_il = typename types::ILTraits<element_type, 2>::type;
    using rank3_il = typename types::ILTraits<element_type, 3>::type;
    using rank4_il = typename types::ILTraits<element_type, 4>::type;

    /// Pull in base class's ctors
    using base_type::base_type;

    /** @brief Allocates a contiguous pointer given @p layout.
     *
     *  @note These methods shadow the function of the same name in the base
     *        class. The intent is to avoid needing to rebind a freshly
     *        allocated buffer when the user already knows it is a Contiguous
     *        buffer.
     *
     *  @param[in] layout The layout of the tensor to allocate. May be passed as
     *                    a unique_ptr or by reference. If passed by reference
     *                    will be copied.
     *
     *  @return A pointer to the newly allocated buffer::Contiguous object.
     */
    ///@{
    contiguous_pointer allocate(const_layout_reference layout) {
        return allocate(layout.clone_as<layout_type>());
    }
    contiguous_pointer allocate(layout_pointer layout) {
        auto p = allocate_(std::move(layout));
        return detail_::static_pointer_cast<contiguous_buffer_type>(p);
    }
    ///@}

    /// Constructs a contiguous buffer from an initializer list
    ///@{
    contiguous_pointer construct(rank0_il il) { return construct_(il); }
    contiguous_pointer construct(rank1_il il) { return construct_(il); }
    contiguous_pointer construct(rank2_il il) { return construct_(il); }
    contiguous_pointer construct(rank3_il il) { return construct_(il); }
    contiguous_pointer construct(rank4_il il) { return construct_(il); }
    ///@}

    /** @brief Constructs a contiguous buffer and sets all elements to @p value.
     *
     *  @param[in] layout The layout of the buffer to allocate. May be passed
     *                    either by unique_ptr or reference. If passed by
     *                    reference will be copied.
     *
     *  @return A pointer to the newly constructed buffer.
     */
    ///@{
    contiguous_pointer construct(const_layout_reference layout,
                                 element_type value) {
        return construct(layout.clone_as<layout_type>(), std::move(value));
    }
    contiguous_pointer construct(layout_pointer layout, element_type value) {
        return construct_(std::move(layout), std::move(value));
    }
    ///@}

protected:
    virtual contiguous_pointer construct_(rank0_il il) = 0;
    virtual contiguous_pointer construct_(rank1_il il) = 0;
    virtual contiguous_pointer construct_(rank2_il il) = 0;
    virtual contiguous_pointer construct_(rank3_il il) = 0;
    virtual contiguous_pointer construct_(rank4_il il) = 0;

    /// To be overridden by the derived class to implement construct
    virtual contiguous_pointer construct_(layout_pointer layout,
                                          element_type value) = 0;
};

} // namespace tensorwrapper::allocator