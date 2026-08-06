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
#include <memory>
#include <tensorwrapper/shape/shape_fwd.hpp>
#include <tensorwrapper/types/class_traits.hpp>
#include <tensorwrapper/types/common_types.hpp>

namespace tensorwrapper::types {

template<>
struct ClassTraits<shape::ShapeBase> : public CommonTypes {
    using shape_base   = shape::ShapeBase;
    using base_pointer = std::unique_ptr<shape_base>;
};

template<>
struct ClassTraits<const shape::ShapeBase> : public CommonTypes {
    using shape_base   = shape::ShapeBase;
    using base_pointer = std::unique_ptr<shape_base>;
};

template<typename Derived>
struct ClassTraits<shape::SmoothCommon<Derived>>
  : public ClassTraits<shape::ShapeBase> {
    using value_type       = Derived;
    using const_value_type = const value_type;
    using reference        = value_type&;
    using const_reference  = const value_type&;
    using pointer          = value_type*;
    using const_pointer    = const value_type*;
    using slice_type       = Derived;
};

template<typename Derived>
struct ClassTraits<const shape::SmoothCommon<Derived>>
  : public ClassTraits<const shape::ShapeBase> {
    using value_type       = Derived;
    using const_value_type = const value_type;
    using reference        = const value_type&;
    using const_reference  = const value_type&;
    using pointer          = const value_type*;
    using const_pointer    = const value_type*;
    using slice_type       = Derived;
};

template<>
struct ClassTraits<shape::Smooth>
  : public ClassTraits<shape::SmoothCommon<shape::Smooth>> {};

template<>
struct ClassTraits<const shape::Smooth>
  : public ClassTraits<const shape::SmoothCommon<shape::Smooth>> {};

template<typename T>
struct ClassTraits<shape::SmoothView<T>>
  : public ClassTraits<shape::SmoothCommon<T>> {
    using smooth_traits = ClassTraits<T>;
    using pimpl_type    = shape::detail_::SmoothViewPIMPL<T>;
    using const_pimpl_type =
      shape::detail_::SmoothViewPIMPL<typename smooth_traits::const_value_type>;
    using pimpl_pointer       = std::unique_ptr<pimpl_type>;
    using const_pimpl_pointer = std::unique_ptr<const_pimpl_type>;
};

} // namespace tensorwrapper::types
