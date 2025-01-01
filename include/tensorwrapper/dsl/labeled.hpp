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
 *  @tparam ObjectType The object we are associating the labels with. Assumed
 *                     to be Tensor or to derive from one of the following:
 *                     ShapeBase, LayoutBase, or BufferBase.
 */
template<typename ObjectType, typename LabelType = std::string>
class Labeled : public utilities::dsl::Term<Labeled<ObjectType, LabelType>> {
private:
    /// Type of *this
    using my_type = Labeled<ObjectType, LabelType>;

    /// Type *this inherits from
    using op_type = utilities::dsl::BinaryOp<my_type, ObjectType, LabelType>;

    /// Is T cv-qualified?
    template<typename T>
    static constexpr bool is_cv_v = !std::is_same_v<std::decay_t<T>, T>;

    /// Is ObjectType cv-qualified?
    static constexpr bool has_cv_object_v = is_cv_v<ObjectType>;

    /// Does *this have a cv-qualified object and T is mutable?
    template<typename T>
    static constexpr bool is_cv_conversion_v = has_cv_object_v && !is_cv_v<T>;

    /// Enables a function when it is being called to convert a const object.
    template<typename T>
    using enable_if_cv_conversion_t = std::enable_if_t<is_cv_conversion_v<T>>;

public:
    /// Type of the object (useful for TMP)
    using object_type = std::decay_t<ObjectType>;

    /// Type of the parsed labels
    using label_type = DummyIndices<LabelType>;

    /// Type of a mutable reference to the labels
    using label_reference = label_type&;

    /// Type of a read-only reference to the labels
    using const_label_reference = const label_type&;

    /// Type of a read-only reference to an object of object_type
    using const_object_reference = const object_type&;

    /// Type of a (possibly) mutable reference to the object
    using object_reference =
      std::conditional_t<has_cv_object_v, const_object_reference, object_type&>;

    /// Type of a read-only pointer to an object of object_type
    using const_object_pointer = const object_type*;

    /// Type of a pointer to a (possibly) mutable object_type object
    using object_pointer =
      std::conditional_t<has_cv_object_v, const_object_pointer, object_type*>;

    /** @brief Creates a Labeled object that does not alias an object or labels.
     *
     *  This ctor is needed because the base classes assume it is present.
     *  Users shouldn't actually need it.
     *
     *  @throw None No throw guarantee.
     */
    Labeled() = default;

    /** @brief Ensures labels are stored correctly.
     *
     *  @tparam ObjectType2 The type of @p object. Must be implicitly
     *                      convertible to @p ObjectType.
     *  @tparam LabelType2 The type of @p labels. Must be implicitly
     *                      convertible to @p LabelType.
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
    template<typename ObjectType2, typename LabelType2>
    Labeled(ObjectType2&& object, LabelType2&& labels) :
      Labeled(std::forward<ObjectType2>(object),
              label_type(std::forward<LabelType2>(labels))) {}

    template<typename ObjectType2>
    Labeled(ObjectType2&& object, label_type labels) :
      m_object_(&object), m_labels_(std::move(labels)) {}

    /** @brief Allows implicit conversion from mutable objects to const objects
     *
     *  @p ObjectType may have cv-qualifiers. This ctor allows Labeled instances
     *  aliasing mutable objects to be used when Labeled instances aliasing
     *  read-only objects are needed.
     *
     *  @tparam ObjectType2 The object type stored in @p input. Must be
     *                      equivalent to `const ObjectType`.
     *  @tparam <anonymous> Used to disable this overload via SFINAE if
     *                      ObjectType2 != `const ObjectType` or if
     */
    template<typename ObjectType2,
             typename = enable_if_cv_conversion_t<ObjectType2>>
    Labeled(const Labeled<ObjectType2, LabelType>& input) :
      Labeled(input.object(), input.labels()) {}

    object_reference object() {
        assert_object_();
        return *m_object_;
    }

    const_object_reference object() const {
        assert_object_();
        return *m_object_;
    }

    label_reference labels() { return m_labels_; }

    const_label_reference labels() const { return m_labels_; }

    /** @brief Assigns a DSL term to *this.
     *
     *  @tparam TermType The type of the expression being assigned to *this.
     *
     *  Under most circumstances execution of the DSL happens when an
     *  expression is assigned to Labeled object. The assignment happens via
     *  this method.
     *
     *  @param[in] other The expression to assign to *this.
     *
     *  @return *this after assigning @p other to *this.
     */
    template<typename TermType>
    my_type& operator=(TermType&& other) {
        // TODO: other should be rolled into a tensor graph object that can be
        //       manipulated at runtime. Parser is then moved to the backend
        PairwiseParser<ObjectType, LabelType> p;
        auto&& [labels, object] =
          p.dispatch(*this, std::forward<TermType>(other));
        object().assign(*object);
        this->labels() = labels;
        return *this;
    }

    bool operator==(const Labeled& rhs) const noexcept {
        return object().are_equal(rhs.object()) && labels() == rhs.labels();
    }

    bool operator!=(const Labeled& rhs) const noexcept {
        return !((*this) == rhs);
    }

private:
    void assert_object_() const {
        if(m_object_ != nullptr) return;
        throw std::runtime_error("Labeled does not contain an object.");
    }

    object_pointer m_object_ = nullptr;
    label_type m_labels_;
};

} // namespace tensorwrapper::dsl