#pragma once
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
#include <iostream>
#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/dsl/pairwise_parser.hpp>
#include <type_traits>
#include <utilities/dsl/dsl.hpp>

namespace tensorwrapper::dsl {

/** @brief Represents an object whose modes are assigned dummy indices.
 *
 *  @tparam ObjectType the type of the object. Assumed to be a class from the
 *                     shape, symmetry, sparsity, layout, buffer, or tensor
 *                     class hierarchies.
 *  @tparam StringType the type used for string literals. Default is
 *                     std::string.
 *
 *  This class is used to promote TensorWrapper objects into the DSL layer.
 *  Users will interact with this class somewhat transparently (usually via
 *  unnamed temporary objects).
 *
 */
template<typename ObjectType, typename StringType = std::string>
class Labeled : public utilities::dsl::Term<Labeled<ObjectType, StringType>> {
private:
    /// Type of *this
    using my_type = Labeled<ObjectType, StringType>;

    /// Type of *this if ObjectType is const
    using const_my_type = Labeled<const std::decay_t<ObjectType>, StringType>;

    /// Type *this inherits from
    using op_type = utilities::dsl::Term<my_type>;

    /// Is T cv-qualified?
    template<typename T>
    static constexpr bool is_cv_v = !std::is_same_v<std::decay_t<T>, T>;

    /// Is ObjectType cv-qualified?
    static constexpr bool has_cv_object_v = is_cv_v<ObjectType>;

    /// Shorthand for type @p T if ObjectType is const, and @p U otherwise
    template<typename T, typename U>
    using if_cv_t = std::conditional_t<has_cv_object_v, T, U>;

    /// Does *this have a cv-qualified object and T is mutable?
    template<typename T>
    static constexpr bool is_cv_conversion_v = has_cv_object_v && !is_cv_v<T>;

    /// Enables a function when it is being called to convert a const object.
    template<typename T>
    using enable_if_cv_conversion_t = std::enable_if_t<is_cv_conversion_v<T>>;

public:
    // -------------------------------------------------------------------------
    // -- Types associated with the object
    // -------------------------------------------------------------------------

    /// Type of the object (useful for TMP)
    using object_type = std::decay_t<ObjectType>;

    /// Type of a read-only reference to an object of object_type
    using const_object_reference = const object_type&;

    /// Type of a (possibly) mutable reference to an object of object_type
    using object_reference = if_cv_t<const_object_reference, object_type&>;

    // -------------------------------------------------------------------------
    // -- Types associated with the labels
    // -------------------------------------------------------------------------

    /// Type of the string literal used for index labels
    using string_type = StringType;

    /// Type of the object managing the parsed index labels
    using label_type = dsl::DummyIndices<string_type>;

    /// Mutable reference to an object of type label_type
    using label_reference = label_type&;

    /// Read-only reference to an object of type label_type
    using const_label_reference = const label_type&;

    /** @brief Associates a set of dummy indices with an object.
     *
     *  @tparam ObjectType2 The type of @p object. Must be implicitly
     *                      convertible to @p ObjectType.
     *  @tparam LabelType2 The type of @p labels. Assumed to be implicitly
     *                     convertible to either StringType or label_type.
     *
     *  It is common for the labels to actually be a string literal, e.g.,
     *  code like `"i,j"`. Type detection for such a type will not match it
     *  to LabelType. We solve this by using this ctor to explicitly convert
     *  @p labels into LabelType before the base class does its TMP.
     *
     *  @param[in] object The object the labels apply to.
     *  @param[in] labels The annotations for the tensor.
     *
     *  @throw std::bad_alloc if converting @p labels to LabelType throws.
     *                        Strong throw guarantee.
     */
    template<typename ObjectType2, typename LabelType>
    Labeled(ObjectType2&& object, LabelType&& labels) :
      m_object_(&object), m_labels_(std::forward<LabelType>(labels)) {}

    /** @brief Allows implicit conversion from mutable objects to const objects
     *
     *  @tparam ObjectType2 The object type stored in @p input. Must be
     *                      equivalent to `const ObjectType`.
     *  @tparam <anonymous> Used to disable this overload via SFINAE if
     *                      ObjectType2 != `const ObjectType` or if
     *                      `ObjectType` is not mutable.
     *
     *  @p ObjectType may have cv-qualifiers. This ctor allows Labeled instances
     *  aliasing mutable objects to be used when Labeled instances aliasing
     *  read-only objects are needed.
     *
     *  @param[in] input The Labeled object to convert.
     */
    template<typename ObjectType2,
             typename = enable_if_cv_conversion_t<ObjectType2>>
    Labeled(const Labeled<ObjectType2, StringType>& input) :
      Labeled(input.object(), input.labels()) {}

    /** @brief Creates a new Labeled object by copying @p other.
     *
     *  The Labeled object created with this ctor will alias the same object
     *  as @p other did, but contain a deep copy of the labels associated with
     *  @p other.
     *
     *  @param[in] other The Labeled object to copy.
     *
     *  @throw std::bad_alloc if there's a problem allocating memory for the
     *                        copy. Strong throw guarantee.
     */
    Labeled(const Labeled& other) = default;

    /** @brief Creates a new Labeled object by taking the state from @p other.
     *
     *  The Labeled object created with this ctor will alias the same object
     *  as @p other did and take ownership of the labels which were previously
     *  associated with @p other.
     *
     *  @param[in,out] other The Labeled object to take the state from. After
     *                       this operation @p other is in a valid, but
     *                       otherwise undefined state.
     *
     *  @throw None No throw guarantee.
     */
    Labeled(Labeled&& other) noexcept = default;

    /** @brief Sets *this equal to @p rhs.
     *
     *  This method can be used as a copy assignment for the object *this
     *  aliases, but it is NOT copy assignment for *this. More specifically
     *  this method will call assign_ to do the actual assignment, which may
     *  result in permutations and or traces being taken of @p rhs before the
     *  assignment happens. Whether permutations/traces occur depends on the
     *  indices of *this.
     *
     *  @note This method is needed because the compiler prefers the
     *        compiler generated version over the function template overload.
     *
     *  @param[in] rhs The object to assign to *this.
     *
     *  @return *this after assigning @p rhs to it and performing any operations
     *          specified by the dummy indices.
     *
     *  @throw ??? Throws if assign_ throws. Same throw guarantee.
     *
     */
    Labeled& operator=(const Labeled& rhs) { return assign_(rhs); };

    /** @brief Sets *this equal to @p rhs.
     *
     *  This method behaves similar to operator=(const Labeled&) except that
     *  the parser has the option of reusing @p rhs in the operation instead
     *  of copying it.
     *
     *  @note This method is needed because the compiler prefers the
     *        compiler generated version over the function template overload.
     *
     *  @param[in,out] rhs The object to assign to *this. After this operation
     *                     @p rhs is in a valid, but otherwise undefined state.
     *
     *  @throw ??? Throws if assign_ throws. Same throw guarantee.
     */
    Labeled& operator=(Labeled&& rhs) { return assign_(std::move(rhs)); }

    /** @brief Assigns a DSL term to *this.
     *
     *  @tparam TermType The type of the expression being assigned to *this.
     *
     *  This method is the generalization of operator=(const Labeled&) to
     *  other leaves of the AST. Like the other operator= methods it is
     *  implemented by calling assign_.
     *
     *  @param[in] other The object containing the AST to assign to *this.
     *
     *  @return *this after assigning @p other to *this.
     *
     *  @throw ??? If assign_ throws. Same throw guarantee.
     */
    template<typename TermType>
    my_type& operator=(TermType&& other) {
        return assign_(std::forward<TermType>(other));
    }

    /** @brief Returns a (possibly) read-only reference to the object.
     *
     *  This method is used to access the object associated with the dummy
     *  indices. The object is mutable if @p ObjectType is a mutable type and
     *  read-only if @p ObjectType is cv-qualified.
     *
     *  @return A reference to the object.
     *
     *  @throw std::runtime_error if *this does not have an object associated
     *                            with it. Strong throw guarantee.
     */
    object_reference object() {
        assert_has_object_();
        return *m_object_;
    }

    /** @brief Returns a read-only reference to the labeled object.
     *
     *  This method is identical to the non-const version except that the
     *  resulting object is guarantee to be read-only. See the description for
     *  the non-const version for more details.
     *
     *  @return A read-only reference to the labeled object.
     *
     *  @throw std::runtime_error if *this does not have an object associated
     *                            with it. Strong throw guarantee.
     */
    const_object_reference object() const {
        assert_has_object_();
        return *m_object_;
    }

    /** @brief The dummy indices associated with the object.
     *
     *  This method is used to retrieve the dummy indices associated with the
     *  object.
     *
     *  @return A mutable reference to the dummy indices.
     *
     *  @throw None No throw guarantee.
     *
     */
    label_reference labels() noexcept { return m_labels_; }

    /** @brief Returns a read-only reference to the dummy labels.
     *
     *  This method is identical to the non-const version, except that the
     *  resulting indices are guaranteed to be read-only. See the description
     *  for the non-const version for more details.
     *
     *  @return A read-only reference to the dummy indices.
     *
     *  @throw None No throw guarantee.
     */
    const_label_reference labels() const noexcept { return m_labels_; }

    /** @brief Does *this have an object?
     *
     *  Under most circumstances *this will be associated with an object. This
     *  method can be used to explicitly test that *this does have an object.
     *
     *  @return True if *this has an object and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool has_object() const noexcept { return static_cast<bool>(m_object_); }

    /** @brief Determines if *this is value equal to @p rhs.
     *
     *  Two Labeled objects are value equal if their labels compare value equal
     *  and if they both:
     *  1. Contain objects which compare (polymorphically) value equal, or
     *  2. Do not contain objects.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const const_my_type& rhs) const noexcept {
        if(has_object() != rhs.has_object()) return false;
        if(labels() != rhs.labels()) return false;
        if(!has_object()) return true;
        return object().are_equal(rhs.object());
    }

    /** @brief Is *this different from @p rhs?
     *
     *  This method simply negates operator==. See operator== for the definition
     *  of value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const const_my_type& rhs) const noexcept {
        return !((*this) == rhs);
    }

private:
    /// Type of a pointer to a read-only  object of object_type
    using const_object_pointer = const object_type*;

    /// Type of a pointer to a (possibly) mutable  object of object_type
    using object_pointer = if_cv_t<const_object_pointer, object_type*>;

    /// Asserts that m_object_ is non-null
    void assert_has_object_() const {
        if(has_object()) return;
        throw std::runtime_error("Object is null. Was it moved from?");
    }

    /// Common implementation for assigning other to *this.
    template<typename TermType>
    Labeled& assign_(TermType&& other) {
        // TODO: other should be rolled into a tensor graph object that can
        //       be manipulated at runtime. Parser is then moved to the backend
        PairwiseParser p;
        p.dispatch(*this, std::forward<TermType>(other));
        return *this;
    }

    /// The object whose modes are indexed.
    object_pointer m_object_ = nullptr;

    /// The dummy indices associated with m_object_
    label_type m_labels_;
};

} // namespace tensorwrapper::dsl