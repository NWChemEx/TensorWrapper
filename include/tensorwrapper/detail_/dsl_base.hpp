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
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/dsl/labeled.hpp>

namespace tensorwrapper::detail_ {

/** @brief Code factorization for objects that are composable via the DSL.
 *
 *  @tparam DerivedType the type of the object which wants to interact with the
 *                      DSL. @p DerivedType is assumed to have a clone method.
 *
 *  This class defines the API parsers of the abstract syntax tree can interact
 *  with to interact with labeled objects generically. Most operations defined
 *  by *this have defaults (which just throw with a "not yet implemented"
 *  error) so that derived classes do not have to override all methods all at
 *  once.
 */
template<typename DerivedType, typename StringType = std::string>
class DSLBase {
public:
    /// Type of the derived class
    using dsl_value_type = DerivedType;

    /// Type of a read-only object of type dsl_value_type
    using dsl_const_value_type = const dsl_value_type;

    /// Type of a reference to an object of type dsl_value_type
    using dsl_reference = dsl_value_type&;

    /// Type of a read-only reference to an object of type dsl_value_type
    using dsl_const_reference = const dsl_value_type&;

    /// Type of a pointer to an object of type dsl_value_type
    using dsl_pointer = std::unique_ptr<dsl_value_type>;

    /// Type used for representing the dummy indices as a string
    using string_type = StringType;

    /// Type of a labeled object
    using labeled_type = dsl::Labeled<dsl_value_type, string_type>;

    /// Type of a labeled read-only object (n.b. labels are mutable)
    using labeled_const_type = dsl::Labeled<dsl_const_value_type, string_type>;

    /// Type of parsed labels
    using label_type = typename labeled_type::label_type;

    /// Type of a read-only reference to a labeled_type object
    using const_labeled_reference = const labeled_const_type&;

    /// Polymorphic no-throw defaulted dtor
    virtual ~DSLBase() noexcept = default;

    /** @brief Associates labels with the modes of *this.
     *
     *  @tparam LabelType The type of @p labels. Assumed to be explicitly
     *                    convertible to label_type.
     *
     *  This method is used to create a labeled object by pairing *this
     *  with the provided labels. The resulting object is capable of being
     *  composed via the DSL.
     *
     *  N.b., the resulting term aliases *this and the user is responsible for
     *  ensuring that *this is not deallocated.
     *
     *  @param[in] labels The indices to associate with the modes of *this.
     *
     *  @return A DSL term pairing *this with @p labels.
     *
     *  @throw None No throw guarantee.
     */
    template<typename LabelType>
    labeled_type operator()(LabelType&& labels) {
        label_type this_labels(std::forward<LabelType>(labels));
        return labeled_type(downcast_(), std::move(this_labels));
    }

    /** @brief Associates labels with the modes of *this.
     *
     *  @tparam LabelType The type of @p labels. Assumed to be explicitly
     *                    convertible to label_type.
     *
     *  This method is the same as the non-const version except that the result
     *  contains a read-only reference to *this.
     *
     *  @param[in] labels The labels to associate with *this.
     *
     *  @return A DSL term pairing *this with @p labels.
     *
     *  @throw None No throw guarantee.
     */
    template<typename LabelType>
    labeled_const_type operator()(LabelType&& labels) const {
        label_type this_labels(std::forward<LabelType>(labels));
        return labeled_const_type(downcast_(), std::move(this_labels));
    }

    // -------------------------------------------------------------------------
    // -- BLAS-Like Operations
    // -------------------------------------------------------------------------

    /** @brief Set this to the result of @p lhs + @p rhs.
     *
     *  This method will overwrite the state of *this with the result of
     *  adding @p lhs to @p rhs.
     *
     *  @param[in] this_labels The labels to associate with the modes of *this.
     *  @param[in] lhs The object to add to @p rhs
     *  @param[in] rhs The object to add to @p lhs.
     *
     *  @return *this after assigning the sum of @p lhs plus @p rhs to *this.
     *
     *  @throws ??? Throws if the derived class's implementation throws. Same
     *              throw guarantee.
     */
    template<typename LabelType>
    dsl_reference addition_assignment(LabelType&& this_labels,
                                      const_labeled_reference lhs,
                                      const_labeled_reference rhs);

    /** @brief Set this to the result of @p lhs - @p rhs.
     *
     *  This method will overwrite the state of *this with the result of
     *  subtracting @p rhs from @p lhs.
     *
     *  @param[in] this_labels The labels to associate with the modes of *this.
     *  @param[in] lhs The object to subtract from.
     *  @param[in] rhs The object to be subtracted.
     *
     *  @return *this after assigning the difference of @p lhs and @p rhs to
     *          *this.
     *
     *  @throws ??? Throws if the derived class's implementation throws. Same
     *              throw guarantee.
     */
    template<typename LabelType>
    dsl_reference subtraction_assignment(LabelType&& this_labels,
                                         const_labeled_reference lhs,
                                         const_labeled_reference rhs);

    /** @brief Set this to the result of @p lhs * @p rhs.
     *
     *  This method will overwrite the state of *this with the result of
     *  multiplying @p lhs with @p rhs. This method is responsible for
     *  element-wise multiplication, contraction, and mixed operations.
     *
     *  @param[in] this_labels The labels to associate with the modes of *this.
     *  @param[in] lhs The object to subtract from.
     *  @param[in] rhs The object to be subtracted.
     *
     *  @return *this after assigning the product of @p lhs and @p rhs to
     *          *this.
     *
     *  @throws ??? Throws if the derived class's implementation throws. Same
     *              throw guarantee.
     */
    template<typename LabelType>
    dsl_reference multiplication_assignment(LabelType&& this_labels,
                                            const_labeled_reference lhs,
                                            const_labeled_reference rhs);

    /** @brief Sets *this to a permutation of @p rhs.
     *
     *  `rhs.labels()` are the dummy indices associated with the modes of the
     *  object in @p rhs and @p this_labels are the dummy indices associated
     *  with the object in *this. This method will permute @p rhs so that the
     *  resulting object's modes are ordered consistently with @p this_labels,
     *  i.e. the permutation is FROM the `rhs.labels()` order TO the
     *  @p this_labels order. This is seemingly backwards when described out,
     *  but consistent with the intent of a DSL expression like
     *  `t("i,j") = x("j,i");` where the intent is to set `t` equal to the
     *  transpose of `x`.
     *
     *  @param[in] this_labels the dummy indices for the modes of *this.
     *  @param[in] rhs The object to permute.
     *
     *  @return *this after setting it equal to a permutation of @p rhs.
     *
     *  @throw std::runtime_error if @p this_labels does not contain the same
     *                            number of indices as *this does modes. Strong
     *                            throw guarantee.
     *  @throw std::runtime_error if @p this_labels contains more dummy indices
     *                            than @p rhs. Strong throw guarantee.
     *  @throw ??? If the derived class's implementation of permute_assignment_
     *             throws. Same throw guarantee.
     */
    template<typename LabelType>
    dsl_reference permute_assignment(LabelType&& this_labels,
                                     const_labeled_reference rhs);

    /** @brief Scales *this by @p scalar.
     *
     *  @tparam ScalarType The type of @p scalar. Assumed to be a floating-
     *                     point type.
     *
     *  This method is responsible for scaling @p rhs by @p scalar and assigning
     *  it to *this.
     *
     *  @note This method is templated on the scalar type to avoid limiting the
     *        API. That said, at present the backend converts @p scalar to
     *        double precision, but we could use a variant or something similar
     *        to avoid this
     */
    template<typename LabelType, typename ScalarType>
    dsl_reference scalar_multiplication(LabelType&& this_labels,
                                        ScalarType&& scalar,
                                        const_labeled_reference rhs);

protected:
    /// Derived class should overwrite to implement addition_assignment
    virtual dsl_reference addition_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
        throw std::runtime_error("Addition assignment NYI");
    }

    /// Derived class should overwrite to implement subtraction_assignment
    virtual dsl_reference subtraction_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
        throw std::runtime_error("Subtraction assignment NYI");
    }

    /// Derived class should overwrite to implement multiplication_assignment
    virtual dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) {
        throw std::runtime_error("Multiplication assignment NYI");
    }

    /// Derived class should overwrite to implement permute_assignment
    virtual dsl_reference permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
        throw std::runtime_error("Permute assignment NYI");
    }

    /// Derived class should overwrite to implement scalar_multiplication
    virtual dsl_reference scalar_multiplication_(label_type this_labels,
                                                 double scalar,
                                                 const_labeled_reference rhs) {
        throw std::runtime_error("Scalar multiplication NYI");
    }

private:
    /// Checks that the dummy indices on an object are consistent with its rank
    void assert_indices_match_rank_(const_labeled_reference other) const {
        const auto rank = other.object().rank();
        const auto n    = other.labels().size();
        if(rank == n) return;
        throw std::runtime_error(
          std::to_string(n) + " dummy indices is incompatible with an object"
                              " with rank " = std::to_string(rank));
    }

    /// Checks that @p output is a subset of @p input
    void assert_is_subset_(const label_type& output,
                           const label_type& input) const {
        if(output.intersection(input).size() < output.unique_index_size())
            throw std::runtime_error(
              "Output indices must be a subset of input indices");
    }

    /// Asserts that @p lhs is a permutation of @p rhs
    void assert_is_permutation_(const label_type& lhs,
                                const label_type& rhs) const {
        if(lhs.is_permutation(rhs)) return;
        throw std::runtime_error(
          "Dummy indices are not related via permutation.");
    }

    /// Wraps getting a mutable reference to the derived class
    decltype(auto) downcast_() { return static_cast<dsl_reference>(*this); }

    /// Wraps getting a read-only reference to the derived class
    decltype(auto) downcast_() const {
        return static_cast<dsl_const_reference>(*this);
    }
};

} // namespace tensorwrapper::detail_

#include "dsl_base.ipp"