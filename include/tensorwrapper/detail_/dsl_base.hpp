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
template<typename DerivedType>
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

    /// Type used for the dummy indices
    using label_type = std::string;

    /// Type of a labeled object
    using labeled_type = dsl::Labeled<dsl_value_type, label_type>;

    /// Type of a labeled read-only object (n.b. labels are mutable)
    using labeled_const_type = dsl::Labeled<dsl_const_value_type, label_type>;

    /// Type of a read-only reference to a labeled_type object
    using const_labeled_reference = const labeled_const_type&;

    /// Polymorphic no-throw defaulted dtor
    virtual ~DSLBase() noexcept = default;

    /** @brief Associates labels with the modes of *this.
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
    labeled_type operator()(label_type labels) {
        return labeled_type(downcast_(), std::move(labels));
    }

    /** @brief Associates labels with the modes of *this.
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
    labeled_const_type operator()(label_type labels) const {
        return labeled_const_type(downcast_(), std::move(labels));
    }

    // -------------------------------------------------------------------------
    // -- BLAS Operations
    // -------------------------------------------------------------------------

    /** @brief Set this to the result of *this + rhs.
     *
     *  This method will overwrite the state of *this with the result of
     *  adding the original state of *this to that of @p rhs. Depending on the
     *  value @p this_labels compared to the labels associated with @p rhs,
     *  it may be a permutation of @p rhs that is added to *this.
     *
     *  @param[in] this_labels The labels to associate with the modes of *this.
     *  @param[in] rhs The object to add into *this.
     *
     *  @throws ??? Throws if the derived class's implementation throws. Same
     *              throw guarantee.
     */
    dsl_reference addition_assignment(label_type this_labels,
                                      const_labeled_reference rhs) {
        return addition_assignment_(std::move(this_labels), rhs);
    }

    /** @brief Returns the result of *this + rhs.
     *
     *  This method is the same as addition_assignment except that the result
     *  is returned in a newly allocated object instead of overwriting *this.
     *
     *  @param[in] this_labels the labels for the modes of *this.
     *  @param[in] rhs The object to add to *this.
     *
     *  @return The object resulting from adding *this to @p rhs.
     *
     *  @throw std::bad_alloc if there is a problem copying *this. Strong throw
     *                        guarantee.
     *  @throw ??? If addition_assignment throws when adding @p rhs to the
     *             copy of *this. Same throw guarantee.
     */
    dsl_pointer addition(label_type this_labels,
                         const_labeled_reference rhs) const {
        auto pthis = downcast_().clone();
        pthis->addition_assignment(std::move(this_labels), rhs);
        return pthis;
    }

    /** @brief Sets *this to a permutation of @p rhs.
     *
     *  `rhs.rhs()` are the dummy indices associated with the modes of the
     *  object in @p rhs and @p this_labels are the dummy indices associated
     *  with the object in *this. This method will permute @p rhs so that the
     *  resulting object's modes are ordered consistently with @p this_labels,
     *  i.e. the permutation is FROM the `rhs.rhs()` order TO the
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
     *  @throw ??? If the derived class's implementation of permute_assignment_
     *             throws. Same throw guarantee.
     */
    dsl_reference permute_assignment(label_type this_labels,
                                     const_labeled_reference rhs) {
        return permute_assignment_(std::move(this_labels), rhs);
    }

    /** @brief Returns a copy of *this obtained by permuting *this.
     *
     *  This method simply calls permute_assignment on a copy of *this. See the
     *  description of permute_assignment for more details.
     *
     *  @param[in] this_labels dummy indices representing the modes of *this in
     *                         its current state.
     *  @param[in] out_labels how the user wants the modes of *this to be
     *                        ordered.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     *  @throw ??? If the derived class's implementation of permute_assignment_
     *             throws. Same throw guarantee.
     */
    dsl_pointer permute(label_type this_labels, label_type out_labels) const {
        auto pthis = downcast_().clone();
        pthis->permute_assignment(std::move(out_labels), (*this)(this_labels));
        return pthis;
    }

protected:
    /// Derived class should overwrite to implement addition_assignment
    virtual dsl_reference addition_assignment_(label_type this_labels,
                                               const_labeled_reference rhs) {
        throw std::runtime_error("Addition assignment NYI");
    }

    /// Derived class should overwrite to implement permute_assignment
    virtual dsl_reference permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
        throw std::runtime_error("Permute assignment NYI");
    }

private:
    /// Wraps getting a mutable reference to the derived class
    DerivedType& downcast_() { return static_cast<DerivedType&>(*this); }

    /// Wraps getting a read-only reference to the derived class
    const DerivedType& downcast_() const {
        return static_cast<const DerivedType&>(*this);
    }
};

} // namespace tensorwrapper::detail_