#pragma once
#include <set>
#include <tensorwrapper/symmetry/relation.hpp>
#include <type_traits>
#include <vector>

namespace tensorwrapper::symmetry {

/** @brief Describes a permutation of a tensor's modes.
 *
 *  This class models a permutation made up of zero or more cycles. Non-trivial
 *  cycles (those which actually swap modes of the tensor) are stored
 *  explicitly all other cycles are stored implicitly.
 */
class Permutation : public Relation {
public:
    /// Pull in types from the base class
    using Relation::base_pointer;
    using Relation::base_type;
    using Relation::mode_index_type;

    using initializer_list = std::initializer_list<mode_index_type>;

    /** @brief Creates an identity permutation.
     *
     *  The default Permutation contains no explicit cycles making it equivalent
     *  to storing only implicit fixed points for an arbitrary rank tensor.
     *  Such a Permutation is equivalent to the identity permutation (i.e.,
     *  do nothing).
     *
     *  @throw No throw guarantee.
     */
    Permutation() = default;

    explicit Permutation(initializer_list il) :
      Permutation(cycle_type(il.begin(), il.end())) {}

    Permutation(initializer_list cycle0, initializer_list cycle1) :
      Permutation(cycle_type(cycle0.begin(), cycle0.end()),
                  cycle_type(cycle1.begin(), cycle1.end())) {}

    /** @brief Determines the minimum rank a tensor must be to apply *this.
     *
     *  Cycles stored in *this are expressed in terms of mode offsets. If for
     *  example a cycle swaps modes 3 and 4 we know that we can only apply such
     *  a permutation to a tensor with a minimum rank of 5 (otherwise it would
     *  not have a mode with offset 4). This method analyzes the cycles stored
     *  in *this and finds the largest mode offset.
     *
     *  @return The maximum mode offset involved in any non-trivial cycle.
     *
     *  @throw None No throw guarantee.
     */
    mode_index_type minimum_rank() const noexcept;

    /** @brief Returns the number of non-trivial cycles in the permutation.
     *
     *  Every permutation can be expressed as a series of one or more cycles.
     *  Cycles with lengths of 0 or 1 are trivial in the sense that they don't
     *  actually move any modes (a cycle of length 0 is a permutation on the
     *  empty set, and a cycle of length 1 preserves the position of its single
     *  member). This method counts the number of non-trivial cycles comprising
     *  *this.
     *
     *  @return The number of non-trivial cycles in the permutation.
     */
    mode_index_type size() const noexcept { return m_cycles_.size(); }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    void swap(Permutation& other) noexcept { m_cycles_.swap(other.m_cycles_); }

    bool operator==(const Permutation& rhs) const noexcept {
        return m_cycles_ == rhs.m_cycles_;
    }

    bool operator!=(const Permutation& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    /// Implements clone by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<Permutation>(*this);
    }

private:
    /// Type used to hold a cycle
    using cycle_type = std::vector<mode_index_type>;

    /// Trait used to determine if @p Args are cycles
    template<typename... Args>
    static constexpr bool are_cycles_v = (std::is_same_v<Args, cycle_type> &&
                                          ...);

    /// Type that enables a function if @p T is a cyclce
    template<typename... Args>
    using enable_if_cycles_t = std::enable_if_t<are_cycles_v<Args...>>;

    /// Type of container holding a set of cycles
    using cycle_container_type = std::set<cycle_type>;

    static auto remove_trivial_cycles_(cycle_container_type input) noexcept {
        /// cppreference.com suggests we can erase using iterators to input
        auto begin = input.begin();
        auto end   = input.end();

        while(begin != end) {
            if(begin->size() < 2) input.erase(begin);
            ++begin;
        }

        return input;
    }

    /// Wraps the provided cycle(s) in a cycle_container
    template<typename... Args,
             typename = enable_if_cycles_t<std::decay_t<Args>...>>
    Permutation(Args&&... args) :
      Permutation(cycle_container_type{std::forward<Args>(args)...}) {}

    explicit Permutation(cycle_container_type cycles) :
      m_cycles_(remove_trivial_cycles_(std::move(cycles))) {
        // TODO: remove trivial cycles
    }

    /// The modes which can be freely permuted among each other
    cycle_container_type m_cycles_;
};

// -----------------------------------------------------------------------------
// -- Out of line inline implementations
// -----------------------------------------------------------------------------

inline Permutation::mode_index_type Permutation::minimum_rank() const noexcept {
    if(m_cycles_.empty()) return mode_index_type(0);

    mode_index_type the_max(1);
    for(const auto& cycle : m_cycles_) {
        the_max =
          std::max(*std::max_element(cycle.begin(), cycle.end()), the_max);
    }
    return the_max;
}

} // namespace tensorwrapper::symmetry
