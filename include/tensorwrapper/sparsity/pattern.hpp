/*
 * Copyright 2024 NWChemEx Community
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
#include <tensorwrapper/detail_/dsl_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>

namespace tensorwrapper::sparsity {

/** @brief Base class for objects describing the sparsity of a tensor. */
class Pattern : public tensorwrapper::detail_::DSLBase<Pattern>,
                public tensorwrapper::detail_::PolymorphicBase<Pattern> {
private:
    /// Type defining the polymorphic API of *this
    using polymorphic_base = tensorwrapper::detail_::PolymorphicBase<Pattern>;

public:
    /// Type used for indexing and offsets
    using size_type = std::size_t;

    /** @brief Creates a pattern for a rank @p rank tensor.
     *
     *  This constructor creates a sparsity pattern for a dense tensor with
     *  @p rank modes.
     *
     *  @param[in] rank The number of modes in the associated tensor.
     *
     *  @throw None No throw guarantee.
     */
    Pattern(size_type rank = 0) noexcept : m_rank_(rank) {}

    /** @brief Provides the rank of the tensor *this assumes.
     *
     *  @return The rank of the tensor *this describes.
     *
     *  @throw None No throw guarantee.
     */
    size_type rank() const noexcept { return m_rank_; }

    /** @brief Determines if *this and @p rhs describe the same sparsity
     *         pattern.
     *
     *  At present the sparsity component only tracks the rank of the tensor so
     *  two Patterns are value equal if they describe tensors with the same
     *  rank.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Pattern& rhs) const noexcept {
        return rank() == rhs.rank();
    }

    /** @brief Is *this different from @p rhs?
     *
     *  This class defines "different" as not value equal. See the description
     *  of operator== for the definition of value equal.
     *
     *  @param[in] rhs The object to compare against
     *
     *  @return False if *this and @p rhs are value equal and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Pattern& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /// Implements clone by calling copy constructor
    typename polymorphic_base::base_pointer clone_() const override {
        return std::make_unique<Pattern>(*this);
    }

    /// Implements are_equal by calling implementation provided by the base
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return are_equal_impl_<Pattern>(rhs);
    }

    /// Implements addition_assignment via permute_assignment
    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;

    /// Implements subtraction_assignment via permute_assignment
    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;

    /// Implements multiplication_assignment via permute_assignment
    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    /// Implements permute_assignment by permuting the extents in @p rhs.
    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

private:
    /// The rank of the tensor associated with *this
    size_type m_rank_;
};

} // namespace tensorwrapper::sparsity
