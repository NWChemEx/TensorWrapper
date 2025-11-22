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
 *  @note Stored cycles will be canonicalized and sorted. "Canonicalized" means
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
     *  The identity permutation for a rank `r` tensor contains `r` fixed
     *  points. When used as a default ctor this will create the identity
     *  permutation for a scalar (rank 0 tensor).
     *
     *  @param[in] rank The rank of the tensor this permutation represents.
     *                  Default is 0.
     *
     *  @throw No throw guarantee.
     */
    explicit Permutation(mode_index_type rank = 0) : m_rank_(rank) {}

    /** @brief Creates a Permutation from "one-line" notation.
     *
     *  One-line notation for a permutation of a rank `r` tensor is an ordered
     *  set of the numbers [0, r) such that i-th number in the set is the
     *  new mode offset of what was the `i`-th mode before the permutation,
     *  e.g., the permutation (1, 0, 3, 2) means that after the permutation
     *  mode 0 is now mode 1, mode 1 is now mode 0, mode 2 is now mode 3, and
     *  mode 3 is now mode 2. In other words, one-line notation shows the new
     *  mode order written in terms of the old mode offsets.
     *
     *  @note If @p il is a trivial cycle it will NOT be explicitly stored.
     *
     *  @throw std::runtime_error if @p il is not a valid one-line
     *                            representation. Strong throw guarantee.
     *
     *  @throw std::bad_alloc if there is a problem allocating the internal
     *                        state. Strong throw guarantee.
     */
    explicit Permutation(cycle_initializer_list il) :
      Permutation(il.size(),
                  parse_one_line_(cycle_type(il.begin(), il.end()))) {}

    /** @brief Creates a Permutation by explicitly specifying the cycles.
     *
     *  @tparam Args the qualified types of @p args. Each type in @p Args is
     *               assumed to be implicitly convertible to cycle_type.
     *
     *  Any arbitrary permutation can be specified by providing the cycles which
     *  comprise it. This ctor takes the rank of the tensor, and a list of one
     *  or more cycles (zero cycles is handled by the identity constructor).
     *  Any mode not appearing in @p cycle0 or @p args is assumed to be a
     *  fixed point.
     *
     *  @param[in] cycle0 The first cycle in the permutation.
     *  @param[in] args The remaining sizeof...(args) cycles in the permutation.
     *
     *  @throw std::runtime_error if a mode appears more than once in a cycle,
     *                            or if more than one cycle contains the same
     *                            mode. Strong throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the internal
     *                        state. Strong throw guarantee.
     */
    template<typename... Args>
    Permutation(mode_index_type rank, cycle_type cycle0, Args&&... args) :
      Permutation(
        rank, cycle_container_type{std::move(cycle0),
                                   cycle_type(std::forward<Args>(args))...}) {}

    // -------------------------------------------------------------------------
    // -- Getters
    // -------------------------------------------------------------------------

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

    /** @brief Permutes the objects in @p input according to *this.
     *
     *  @tparam T The type of a container-like object. It must support size(),
     *            and operator[].
     *
     *  @param[in] input The object to permute.
     *
     *  @return A copy of @p input with its elements permuted according to
     *          *this.
     *
     *  @throw std::runtime_error if the size of @p input does not match the
     *                         rank of *this. Strong throw guarantee.
     */
    template<typename T>
    T apply(T input) const {
        if(input.size() != m_rank_)
            throw std::runtime_error(
              "Input size does not match permutation rank");
        for(const auto& cycle : m_cycles_) {
            if(cycle.size() < 2) continue;
            T buffer = input;
            for(std::size_t i = 0; i < cycle.size(); ++i) {
                auto from = cycle[i];
                auto to   = cycle[(i + 1) % cycle.size()];
                input[to] = buffer[from];
            }
        }
        return input;
    }

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
    void swap(Permutation& other) noexcept {
        m_cycles_.swap(other.m_cycles_);
        std::swap(m_rank_, other.m_rank_);
    }

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
        return std::tie(m_rank_, m_cycles_) ==
               std::tie(rhs.m_rank_, rhs.m_cycles_);
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

    /// If *this has no explicit cycles it is an identity permutation
    bool is_identity_() const noexcept override { return size() == 0; }

    /// Implements are_equal by using implementation provided by the base class.
    bool are_equal_(const_base_reference other) const noexcept override {
        return are_equal_impl_<Permutation>(other);
    }

    /// Implements rank by returning the stored rank
    mode_index_type rank_() const noexcept override { return m_rank_; }

private:
    /// Type of container holding a set of cycles
    using cycle_container_type = std::set<cycle_type>;

    cycle_container_type parse_one_line_(const cycle_type& one_line) const;

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
    Permutation(mode_index_type rank, cycle_container_type cycles) :
      m_cycles_(remove_trivial_cycles_(std::move(cycles))), m_rank_(rank) {
        for(const auto& x : m_cycles_)
            for(auto xi : x)
                if(xi >= m_rank_)
                    throw std::runtime_error(
                      "Offset is inconsistent with rank");
    }

    /// The modes which can be freely permuted among each other
    cycle_container_type m_cycles_;

    /// The overall rank of the tensor
    mode_index_type m_rank_;
};

} // namespace tensorwrapper::symmetry
