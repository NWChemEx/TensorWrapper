#pragma once
#include <tensorwrapper/layout/tiled.hpp>

namespace tensorwrapper::layout {

/** @brief Specializes a tiled layout to when there's a single tile.
 *
 *  Non-distributed tensors typically have no tiling structure. This class
 *  makes it easier to define a tiled layout when there's only a single tile.
 */
class MonoTile : public Tiled {
private:
    /// Type *this derives from
    using base_type = Tiled;

public:
    /// Pull in base class's types
    using base_type::layout_pointer;
    using base_type::size_type;

    /// Reuse base class's ctors
    using base_type::base_type;

protected:
    /// Implements clone by calling copy ctor
    layout_pointer clone_() const override {
        return std::make_unique<MonoTile>(*this);
    }

    /// Hard-codes tile_size_ to return 1
    size_type tile_size_() const noexcept override { return 1; }

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const layout_base& rhs) const noexcept override {
        return are_equal_impl_<MonoTile>(rhs);
    }
};

} // namespace tensorwrapper::layout
