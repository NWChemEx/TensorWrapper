#pragma once
#include <set>
#include <tensorwrapper/symmetry/relation.hpp>
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

    /// Type used to hold a cycle
    using cycle_type = std::vector<mode_index_type>;

    /// Type of an initializer list for a cycle
    using cycle_initializer_list = std::initializer_list<mode_index_type>;

    /// Type of an initializer list for a set of cycles
    using cycle_set_initializer_list =
      std::initializer_list<cycle_initializer_list>;

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

    explicit Permutation(cycle_initializer_list il) :
      Permutation(cycle_container_type{cycle_type(il.begin(), il.end())}) {}

    explicit Permutation(cycle_set_initializer_list cycles);

    // -------------------------------------------------------------------------
    // -- Getters
    // -------------------------------------------------------------------------

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

    /** @brief Obtains the @p i -th non-trivial cycle in *this.
     *
     *  @param[in] i The offset of the requested cycles. Must be in the range
     *               [0, size()).
     *
     *  @return The requested cycle.
     *
     *  @throw None This method does not throw if @p i is invalid. Use `at` if
     *              you would like bounds checking. No throw guarantee.
     */
    cycle_type operator[](mode_index_type i) const noexcept;

    /** @brief Obtains the @p i -th non-trivial cycle in *this.
     *
     *  This method behaves the same as operator[] except that it first checks
     *  that @p i is in bounds.
     *
     *  @param[in] i The offset of the requested cycles. Must be in the range
     *               [0, size()).
     *
     *  @return The requested cycle.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, size()). Strong
     *                           throw guarantee.
     */
    cycle_type at(mode_index_type i) const {
        valid_offset_(i);
        return (*this)[i];
    }

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
    /// Type of container holding a set of cycles
    using cycle_container_type = std::set<cycle_type>;

    void valid_offset_(mode_index_type i) const;

    /// Verifies that @p cycle does not contain repeat elements
    static void is_valid_cycle_(cycle_type cycle);

    /// Verifies that @p input is a valid set of cycles
    static void verify_valid_cycle_set_(const cycle_container_type& cycles);

    /// Cyclically permutes @p cycle so the lowest mode is first
    static cycle_type canonicalize_cycle_(cycle_type cycle);

    /// Removes cycles of length less than 2
    static cycle_container_type remove_trivial_cycles_(
      cycle_container_type input);

    /// Primary ctor for the class. All others dispatch here
    explicit Permutation(cycle_container_type cycles) :
      m_cycles_(remove_trivial_cycles_(std::move(cycles))) {}

    /// The modes which can be freely permuted among each other
    cycle_container_type m_cycles_;
};

} // namespace tensorwrapper::symmetry
