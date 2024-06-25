#pragma once
#include <set>
#include <tensorwrapper/symmetry/relation.hpp>

namespace tensorwrapper::symmetry {

/** @brief A permutation group which leaves the sign of the elements unchanged
 *
 *  Models a set of permutations which can be applied to a tensor's modes
 *  without changing the value of the tensor.
 */
class Symmetric : public Relation {
public:
    /// Pull in types from the base class
    using Relation::base_pointer;
    using Relation::base_type;
    using Relation::mode_index_type;

    /** @brief Creates a (symmetric) identity permutation.
     *
     *
     */
    Symmetric() noexcept = default;

    template<typename... Args>
    explicit Symmetric(mode_index_type m0, Args&&... args) :
      m_modes_{m0, std::forward<Args>(args)...} {}

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    void swap(Symmetric& other) noexcept { m_modes_.swap(other.m_modes_); }

    bool operator==(const Symmetric& rhs) const noexcept {
        return m_modes_ == rhs.m_modes_;
    }

    bool operator!=(const Symmetric& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    /// Implements clone by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<Symmetric>(*this);
    }

private:
    /// Type used to hold the modes
    using mode_container_type = std::set<mode_index_type>;

    /// The modes which can be freely permuted among each other
    mode_container_type m_modes_;
};

} // namespace tensorwrapper::symmetry
