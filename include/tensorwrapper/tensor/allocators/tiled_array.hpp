#pragma once
#include "tensorwrapper/tensor/allocators/allocator.hpp"

namespace tensorwrapper::tensor::allocator {

namespace ta {

enum class Storage { Core };

enum class Tiling { OneBigTile, SingleElementTile };

enum class Distribution { Replicated, Distributed };

} // namespace ta

template<typename FieldType>
class TiledArrayAllocator : public Allocator<FieldType> {
    using base_type = Allocator<FieldType>;
    using my_type   = TiledArrayAllocator;

public:
    using allocator_ptr         = typename base_type::allocator_ptr;
    using runtime_reference     = typename base_type::runtime_reference;
    using scalar_populator_type = typename base_type::scalar_populator_type;
    using value_type            = typename base_type::value_type;
    using shape_type            = typename base_type::shape_type;

    explicit TiledArrayAllocator(
      ta::Storage storage   = ta::Storage::Core,
      ta::Tiling tiling     = ta::Tiling::OneBigTile,
      ta::Distribution dist = ta::Distribution::Replicated,
      runtime_reference rt  = TA::get_default_world()) :
      base_type(rt), storage_(storage), tiling_(tiling), dist_(dist){};

    ~TiledArrayAllocator() noexcept = default;

    // Getters
    inline auto storage() const { return storage_; }
    inline auto tiling() const { return tiling_; }
    inline auto dist() const { return dist_; }

    bool operator==(const my_type& other) const {
        return storage_ == other.storage_ and tiling_ == other.tiling_ and
               dist_ == other.dist_;
    }

    bool operator!=(const my_type& other) const {
        return not((*this) == other);
    }

private:
    void hash_(tensorwrapper::detail_::Hasher& h) const override;
    allocator_ptr clone_() const override;
    value_type allocate_(const scalar_populator_type& fxn,
                         const shape_type& shape) const override;
    bool is_equal_(const base_type& rhs) const noexcept override;

    ta::Storage storage_;
    ta::Tiling tiling_;
    ta::Distribution dist_;
};

extern template class TiledArrayAllocator<field::Scalar>;
extern template class TiledArrayAllocator<field::Tensor>;

} // namespace tensorwrapper::tensor::allocator
