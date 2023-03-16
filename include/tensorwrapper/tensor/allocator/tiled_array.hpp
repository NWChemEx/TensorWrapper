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
#include "tensorwrapper/tensor/allocator/allocator_class.hpp"

namespace tensorwrapper::tensor::allocator {

namespace ta {

enum class Storage { Core };

enum class Distribution { Replicated, Distributed };

} // namespace ta

template<typename FieldType>
class TiledArrayAllocator : public Allocator<FieldType> {
    using base_type = Allocator<FieldType>;
    using my_type   = TiledArrayAllocator;

public:
    using allocator_ptr          = typename base_type::allocator_ptr;
    using runtime_type           = typename base_type::runtime_type;
    using runtime_reference      = typename base_type::runtime_reference;
    using tile_populator_type    = typename base_type::tile_populator_type;
    using element_populator_type = typename base_type::element_populator_type;
    using value_type             = typename base_type::value_type;
    using value_pointer          = typename base_type::value_pointer;
    using shape_type             = typename base_type::shape_type;

    explicit TiledArrayAllocator(
      ta::Storage storage   = ta::Storage::Core,
      ta::Distribution dist = ta::Distribution::Replicated) :
      storage_(storage), dist_(dist){};

    ~TiledArrayAllocator() noexcept = default;

    // Getters
    inline auto storage() const { return storage_; }
    inline auto dist() const { return dist_; }

    bool operator==(const my_type& other) const {
        return storage_ == other.storage_ and dist_ == other.dist_;
    }

    bool operator!=(const my_type& other) const {
        return not((*this) == other);
    }

private:
    allocator_ptr clone_() const override;
    value_pointer allocate_(const tile_populator_type& fxn,
                            const shape_type& shape) const override;
    value_pointer allocate_(const element_populator_type& fxn,
                            const shape_type& shape) const override;
    value_pointer reallocate_(const value_type&,
                              const shape_type&) const override;
    bool is_equal_(const base_type& rhs) const noexcept override;

    ta::Storage storage_;
    ta::Distribution dist_;
};

extern template class TiledArrayAllocator<field::Scalar>;
extern template class TiledArrayAllocator<field::Tensor>;

} // namespace tensorwrapper::tensor::allocator
