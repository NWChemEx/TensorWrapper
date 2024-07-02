#pragma once
#include <set>
#include <tensorwrapper/symmetry/operation.hpp>
#include <vector>

namespace tensorwrapper::symmetry {

/** @brief Describes a permutation of a tensor's modes.
 *
 *  This class models a permutation made up of zero or more cycles. Non-trivial
 *  cycles (those which actually swap modes of the tensor) are stored
 *  explicitly all other cycles are stored implicitly.
 *
 *  @note Stored cycles will be canonicalized and sorted. "Canoicalized" means
 *        that each cycle will be cyclically permuted until the smallest
 *        element is first), e.g., the cycle 231 will be stored as 123. Sorting
 *        will be done lexicographically, e.g., the cycle 012 will come before
 *        the cycle 345.
 */
class Permutation : public Operation {
public:
    /// Pull in types from the base class
    using Operation::base_pointer;
    using Operation::base_type;
    using Operation::const_base_reference;
    using Operation::mode_index_type;

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

    /** @brief Creates a Permutation containing a single cycle.
     *
     *  Many permutations involve a single cycle. For convenience this ctor has
     *  been defined so that the user can construct the resulting Permutation
     *  object with by only providing a single initialization list, i.e., no
     *  need to do something like `Permutation p123{{1, 2, 3}};`. Ultimately,
     *  this ctor dispatches to `Permutation(cycle_set_initializer_list)`.
     *
     *  @note If @p il is a trivial cycle it will NOT be explicitly stored.
     *
     *  @param[in] il The modes involved in the cycle. Modes need not be in
     *                canonical order.
     *
     *  @throw std::runtime_error if a mode appears more than once in il. Strong
     *                            throw guarantee.
     *
     *  @throw std::bad_alloc if there is a problem allocating the internal
     *                        state. Strong throw guarantee.
     */
    explicit Permutation(cycle_initializer_list il) :
      Permutation(cycle_container_type{cycle_type(il.begin(), il.end())}) {}

    /** @brief Creates a Permutation by explicitly specifying the cycles.
     *
     *  Any arbitrary permutation can be specified by providing the cycles which
     *  comprise it. This ctor takes a list of cycles (each of which is a list
     *  of modes) and creates the resulting Permutation.
     *
     *  @param[in] cycles The cycles comprising the permutation.
     *
     *  @throw std::runtime_error if a mode appears more than once in a cycle,
     *                            or if more than one cycle contains the same
     *                            mode. Strong throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the internal
     *                        state. Strong throw guarantee.
     */
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
     *  @return A copy of the requested cycle.
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
     *
     *  @throw None No throw guarantee.
     */
    mode_index_type size() const noexcept { return m_cycles_.size(); }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Exchanges the state in *this with the state in @p other.
     *
     *  @param[in,out] other The object to swap state with. After this operation
     *                       @p other will contain the state which was
     *                       previously in *this.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Permutation& other) noexcept { m_cycles_.swap(other.m_cycles_); }

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two Permutation objects are the same if they contain the same number of
     *  explicit cycles and if the i-th explicit cycle of the one is equal to
     *  the i-th explicit cycle of the other. Notably this definition does not
     *  account for different implicit cycles, e.g.,
     * `Permutation{{0},{1,2},{3}}` is considered the same as
     * `Permutation{{0},{1,2}}`.
     *
     *  @param[in] rhs The Permutation to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Permutation& rhs) const noexcept {
        return m_cycles_ == rhs.m_cycles_;
    }

    /** @brief Is *this different than @p rhs?
     *
     *  The Permutation class defines different as "not value equal". Hence this
     *  method simply negates operator==. See the documentation for operator==
     *  for the definition of value equal.
     *
     *  @param[in] rhs The Permutation to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee
     */
    bool operator!=(const Permutation& rhs) const noexcept {
        return !(*this == rhs);
    }

protected:
    /// Implements clone by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<Permutation>(*this);
    }

    /// Implements are_equal by using implementation provided by the base class.
    bool are_equal_(const_base_reference other) const noexcept override {
        return are_equal_impl_<Permutation>(other);
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
