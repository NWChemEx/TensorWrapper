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
#include <wtf/wtf.hpp>

namespace tensorwrapper::allocator {

/** @brief Allocator that can create Contiguous buffers.
 *
 *  @tparam FloatType Type of the elements in the contiguous buffer.
 */
class Contiguous : public Replicated {
private:
    /// Type of *this
    using my_type = Contiguous;

    /// Type *this derives from
    using base_type = Replicated;

public:
    /// Pull in base types
    ///@{
    using base_type::buffer_base_pointer;
    using base_type::const_layout_reference;
    using base_type::layout_pointer;
    ///@}

    /// Types associated with the buffer *this makes
    using buffer_type            = buffer::Contiguous;
    using buffer_reference       = buffer_type&;
    using const_buffer_reference = const buffer_type&;
    using buffer_pointer         = std::unique_ptr<buffer_type>;

    using size_type = std::size_t;

    /// Type of initializer lists
    template<typename element_type>
    using rank0_il = typename types::ILTraits<element_type, 0>::type;

    template<typename element_type>
    using rank1_il = typename types::ILTraits<element_type, 1>::type;

    template<typename element_type>
    using rank2_il = typename types::ILTraits<element_type, 2>::type;

    template<typename element_type>
    using rank3_il = typename types::ILTraits<element_type, 3>::type;

    template<typename element_type>
    using rank4_il = typename types::ILTraits<element_type, 4>::type;

    /// Pull in base class's ctors
    using base_type::base_type;

    explicit Contiguous(runtime_view_reference runtime) : base_type(runtime) {}

    /** @brief Determines if @p buffer can be rebound as a Contiguous buffer.
     *
     *  Rebinding a buffer allows the same memory to be viewed as a (possibly)
     *  different type of buffer.
     *
     *  @param[in] buffer The tensor we are attempting to rebind.
     *
     *  @return True if @p buffer can be rebound to the type of buffer
     *          associated with this allocator and false otherwise.
     *
     *  @throw None No throw guarantee
     */
    static bool can_rebind(const_buffer_base_reference buffer);

    /** @brief Rebinds a buffer to the same type as *this.
     *
     *  This method will convert @p buffer into a buffer which could have been
     *  allocated by *this. If @p buffer was allocated as such a buffer already,
     *  then this method is simply a downcast.
     *
     *  @param[in] buffer The buffer to rebind.
     *
     *  @return A mutable reference to @p buffer viewed as a buffer that could
     *          have been allocated by *this.
     *
     *  @throw std::runtime_error if can_rebind(buffer) is false. Strong throw
     *                            guarantee.
     */
    static buffer_reference rebind(buffer_base_reference buffer);
    static const_buffer_reference rebind(const_buffer_base_reference buffer);

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
    buffer_pointer allocate(const_layout_reference layout) {
        return allocate(layout.clone_as<layout_type>());
    }
    buffer_pointer allocate(layout_pointer layout) {
        auto p = allocate_(std::move(layout));
        return detail_::static_pointer_cast<buffer_type>(p);
    }
    ///@}

    /// Constructs a contiguous buffer from an initializer list
    ///@{
    template<typename T>
    buffer_pointer construct(rank0_il<T> il) {
        return il_construct_(il);
    }

    template<typename T>
    buffer_pointer construct(rank1_il<T> il) {
        return il_construct_(il);
    }

    template<typename T>
    buffer_pointer construct(rank2_il<T> il) {
        return il_construct_(il);
    }

    template<typename T>
    buffer_pointer construct(rank3_il<T> il) {
        return il_construct_(il);
    }

    template<typename T>
    buffer_pointer construct(rank4_il<T> il) {
        return il_construct_(il);
    }
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
    template<typename ElementType>
    buffer_pointer construct(const_layout_reference layout, ElementType value) {
        return construct(layout.clone_as<layout_type>(), std::move(value));
    }

    template<typename ElementType>
    buffer_pointer construct(layout_pointer layout, ElementType value) {
        return construct_(std::move(layout), wtf::fp::make_float(value));
    }
    ///@}

protected:
    buffer_base_pointer allocate_(layout_pointer playout) override;

    /// To be overridden by the derived class to implement construct
    virtual buffer_pointer construct_(layout_pointer layout,
                                      wtf::fp::Float value);

    base_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }

    /// Implements are_equal, by deferring to the base's operator==
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return base_type::template are_equal_impl_<my_type>(rhs);
    }

private:
    layout_pointer layout_from_extents_(const std::vector<size_type>& extents);

    template<typename ILType>
    buffer_pointer il_construct_(ILType il) {
        throw std::runtime_error("Fix me!");
        // auto [extents, data] = detail_::unwrap_il(il);
        // auto pbuffer         = this->allocate(layout_from_extents_(extents));
        // auto& buffer_down    = rebind(*pbuffer);
        // buffer_down.copy(data);
        // return pbuffer;
    }
};

} // namespace tensorwrapper::allocator
