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
#include "smooth_view_pimpl.hpp"
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::shape::detail_ {

/** @brief  Implements SmoothView by wrapping a Smooth object.
 *
 *  A common scenario that occurs is that we need to use an actual Smooth
 *  object as if it were a SmoothView object. This class implements a SmoothView
 *  by wrapping a pointer to an actual Smooth object. All member functions of
 *  the Smooth object are simply forwarded through the SmoothView API.
 *
 *  @tparam SmoothType the type of Smooth object *this is acting like a view of.
 */
template<typename SmoothType>
class SmoothAlias : public SmoothViewPIMPL<SmoothType> {
private:
    /// Actual type *this inherits from
    using my_base = SmoothViewPIMPL<SmoothType>;

    /// Template type parameter with const-qualifier removed
    using value_type = std::decay_t<SmoothType>;

    /// Type of a SmoothAlias aliasing a read-only Smooth object.
    using const_my_type = SmoothAlias<const value_type>;

public:
    /// Pull in bases's types
    ///@{
    using base_type        = typename my_base::base_type;
    using base_pointer     = typename my_base::base_pointer;
    using parent_type      = typename my_base::parent_type;
    using smooth_pointer   = typename parent_type::smooth_traits::pointer;
    using smooth_reference = typename parent_type::smooth_reference;
    using rank_type        = typename my_base::rank_type;
    using size_type        = typename my_base::size_type;
    using typename my_base::const_smooth_view_pimpl_pointer;
    ///@}

    /// Aliases @p shape
    explicit SmoothAlias(smooth_reference shape) : m_pshape_(&shape) {}

protected:
    /// Implemented by calling deep copy ctor
    base_pointer clone_() const override {
        return std::make_unique<SmoothAlias>(*this);
    }

    /// These just call shape's member function with the same name
    ///@{
    rank_type extent_(size_type i) const override { return shape_().extent(i); }
    rank_type rank_() const noexcept override { return shape_().rank(); }
    size_type size_() const noexcept override { return shape_().size(); }
    ///@}

    /// Implemented by by passing const reference of the shape *this aliases
    const_smooth_view_pimpl_pointer as_const_() const override {
        return std::make_unique<const_my_type>(*m_pshape_);
    }

private:
    /// Shortens the keystrokes for dereferencing m_pshape_
    decltype(auto) shape_() const { return *m_pshape_; }

    /// The Smooth object we are aliasing.
    smooth_pointer m_pshape_;
};

} // namespace tensorwrapper::shape::detail_
