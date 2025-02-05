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
#include <string>
#include <utilities/containers/indexable_container_base.hpp>
#include <utilities/strings/string_tools.hpp>
#include <vector>

namespace tensorwrapper::dsl {

/** @brief Class to help deal with string-based dummy indices.
 *
 *  @tparam StringType The type used to hold the dummy indices.
 *
 *  The DSL calls for the user to label each mode of a tensor (or shape, or
 *  buffer, or...) with a dummy index. This is usually done by providing a
 *  compile time literal like `"i,j,k"`, which would label the modes of a rank
 *  3 tensor such that mode 0 is assigned dummy index `"i"`, mode 1 is assigned
 *  dummy index `"j"`, and mode 2 is assigned dummy index `"k"`. While strings
 *  are nice for the user, they're less nice for the developer. This class maps
 *  the string the user provided to an ordered sets of objects. The developer
 *  can then request common dummy index manipulations like set difference or
 *  permutation and let *this worry about the string manipulations.
 *
 *  This class defines the string to dummy index conventions used throughout the
 *  TensorWrapper library, namely:
 *
 *  - Dummy indices are separated by commas, i.e., `"i,jk,l"` defines three
 *    indices such that mode 0 is labeled by `"i"`, mode 1 by `"jk"`, and mode
 *    2 by `"l"`.
 *  - Dummy indices can be multiple characters (see previous example)
 *  - Dummy indices are case-sensitive, i.e., `"i,J"` and `"i,j"` result in
 *    different dummy indices for mode 1.
 *  - Spaces are assumed to be for the user's clarity and are stripped prior
 *    to spliting i.e., `"i, j"` and `"i,j"` are the same set of indices. This
 *    also means `"my index,k"` will define a dummy index `"myindex"` for
 *    mode 0.
 */
template<typename StringType>
class DummyIndices
  : public utilities::IndexableContainerBase<DummyIndices<StringType>> {
private:
    /// Type of *this
    using my_type = DummyIndices<StringType>;

    /// Type *this derives from
    using base_type = utilities::IndexableContainerBase<my_type>;

public:
    /// Type used to hold the string representation of the dummy indices
    using value_type = StringType;

    /// Type of a mutable reference to a value_type
    using reference = value_type&;

    /// Type of a read-only reference to a value_type object
    using const_reference = const value_type&;

    /// Type of the string representation after splitting on commas
    using split_string_type = std::vector<value_type>;

    /// Type used for offsets
    using size_type = typename split_string_type::size_type;

    /// Type used for returning ordered sets of size_type objects
    using offset_vector = std::vector<size_type>;

    /** @brief Creates an object with no dummy indices.
     *
     *  Default constructed DummyIndices objects behave like they contain the
     *  dummy indices for a scalar.
     *
     *  @throw None No throw guarantee.
     */
    DummyIndices() = default;

    /** @brief Constructs a DummyIndices object by parsing a string.
     *
     *  We assume that DummyIndices objects will be created directly from user
     *  input and that user input will be in a type implicitly convertible to
     *  `const_reference`. Under these assumptions, this ctor is the main user-
     *  facing ctor for the class. This ctor will first remove spaces in
     *  @p dummy_indices and then split the space-less string on commas.
     *  Finally, it will verify that the resulting vector of dummy indices has
     *  non-empty elements.
     *
     *  @param[in] dummy_indices The string used to initialize *this.
     *
     *  @throw std::runtime_error if @p dummy_indices contains one or more
     *                            commas and if after splitting on the commas
     *                            one or more of the resulting dummy indices is
     *                            empty.
     */
    explicit DummyIndices(const_reference dummy_indices) :
      DummyIndices(
        utilities::strings::split_string(remove_spaces_(dummy_indices), ",")) {}

    /// Main ctor for setting the value, throws if any index is empty
    explicit DummyIndices(split_string_type split_dummy_indices) :
      m_dummy_indices_(std::move(split_dummy_indices)) {
        for(const auto& x : m_dummy_indices_)
            if(x.empty())
                throw std::runtime_error(
                  "Dummy index is not allowed to be empty");
    }

    /** @brief Determines the number of unique indices in *this.
     *
     *  A dummy index can be repeated if it is going to be summed over. This
     *  method analyzes the indices in *this and determines how many of them
     *  are unique.
     *
     *  @return The number of indices which appear only once in *this.
     *
     *  @throw std::bad_alloc if the temporary container can not be allocated.
     *                        Strong throw guarantee.
     */
    size_type unique_index_size() const {
        std::set<value_type> temp(this->begin(), this->end());
        return temp.size();
    }

    /** @brief Does *this have repeated indices?
     *
     *  This method is used to determine if *this contains any index that
     *  appears more than once.
     *
     *  @return True if *this contains a repeated index and false otherwise.
     *
     *  @throw std::bad_alloc if the internal call to unique_index_size()
     *                        throws. Strong throw guarantee.
     */
    bool has_repeated_indices() const {
        return unique_index_size() != this->size();
    }

    /** @brief Determines if *this is a permutation of @p other.
     *
     *  *this is a permutation of @p other if both *this and @p other contain
     *  the same number of dummy indices and if each unique index in *this
     *  appears the same number of times in *this and @p other.
     *
     *  @param[in] other The set of dummy indices to compare against.
     *
     *  @return True if *this is a permutation of @p other and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool is_permutation(const DummyIndices& other) const noexcept {
        // Must be same size
        if(this->size() != other.size()) return false;

        // Each index in *this must show up the same number of times in other
        for(const auto& index : *this) {
            if(count(index) != other.count(index)) return false;
        }
        return true;
    }

    /** @brief Is a thruple of DummyIndices consistent with a pure element-wise
     *         product?
     *
     *  In generalized Einstein notation a pure element-wise (also commonly
     *  termed Hadamard) product is denoted by *this, @p lhs, and @p rhs
     *  having the same ordered set of dummy indices, up to permutation.
     *  Additionally, the dummy indices associated with any given tensor may
     *  not include a repeated index.
     *
     *  @param[in] lhs The dummy indices associated with the tensor to the
     *                 left of the times operator.
     *  @param[in] rhs The dummy indices associated with the tensor to the
     *                 right of the times operator.
     *
     *  @return True If the dummy indices given by *this, @p lhs, and @p rhs
     *          are consistent with a purely element-wise product of the tensors
     *          that @p lhs and @p rhs label.
     *
     *  @throw None No throw guarantee.
     */
    bool is_hadamard_product(const DummyIndices& lhs,
                             const DummyIndices& rhs) const noexcept;

    /** @brief Does a thruple of DummyIndices indicate a product is a pure
     *         contraction?
     *
     *  In generalized Einstein notation a pure contraction is an operation
     *  where indices common to @p lhs and @p rhs are summed over and do NOT
     *  appear in the result, i.e., *this. Additionally, we stipulate that
     *  there must be at least one index summed over (if no index is summed over
     *  the operation is a pure direct-product).
     *
     *  @param[in] lhs The dummy indices associated with the tensor to the
     *                 left of the times operator.
     *  @param[in] rhs The dummy indices associated with the tensor to the
     *                 right of the times operator.
     *
     *  @return True if the indices associated with *this, @p lhs, and @p rhs
     *          are consistent with a contraction and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool is_contraction(const DummyIndices& lhs,
                        const DummyIndices& rhs) const noexcept;

    /** @brief Computes the permutation needed to convert *this into @p other.
     *
     *  Each DummyIndices object is viewed as an ordered set of objects. If
     *  two DummyIndices objects contain the same objects, but in a different
     *  order, we can convert either object into the other by permuting it.
     *  This method computes the permutation needed to change *this into
     *  @p other. More specifically the result of this method is a vector
     *  of length `size()` such that the `i`-th element is the offset of
     *  `(*this)[i]` in @p other, i.e., if `x` is the return then
     *  `other[x[i]] ==  (*this)[i]`.
     *
     *  @param[in] other The order we want to permute *this to.
     *
     *  @return A vector such that the i-th element is the offset of
     *          `(*this)[i]` in @p other.
     *
     *  @throw std::runtime_error if *this and @p other do not have the same
     *                            size, or if either *this or @p other have
     *                            repeated indices, or if an index in *this
     *                            does not appear in @p other. Strong throw
     *                            guarantee in each case.
     *  @throw std::bad_alloc if there is a problem allocating the return.
     *                        Strong throw guarantee.
     */
    offset_vector permutation(const DummyIndices& other) const {
        if(this->size() != other.size())
            throw std::runtime_error("Must have same number of dummy indices.");

        if(has_repeated_indices() || other.has_repeated_indices())
            throw std::runtime_error("Must contain unique dummy indices.");

        offset_vector rv;
        for(const auto& index : *this) {
            auto indices = other.find(index);
            if(indices.empty())
                throw std::runtime_error("Dummy index not found in other");
            rv.push_back(indices[0]);
        }
        return rv;
    }

    /** @brief Finds the offset of @p index_to_find in *this.
     *
     *  This method can be used to determine which modes the dummy index
     *  @p index_to_find maps to. If @p index_to_find does not appear in *this
     *  the result is empty. If @p index_to_find appears more than once the
     *  result will contain the offset for each appearance.
     *
     *  @param[in] index_to_find The dummy index to determine the offset of.
     *
     *  @return A container whose elements are the offsets of @p index_to_find
     *          in *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the return.
     *                        Strong throw guarantee.
     */
    offset_vector find(const_reference index_to_find) const {
        offset_vector rv;
        for(size_type i = 0; i < this->size(); ++i)
            if(m_dummy_indices_[i] == index_to_find) rv.push_back(i);
        return rv;
    }

    /** @brief Determines how many times @p index_to_find occurs in *this.
     *
     *  @param[in] index_to_find The dummy index to find.
     *
     *  @return The number of times dummy index occurs in *this.
     *
     *  @throw None No throw guarantee.
     */
    size_type count(const_reference index_to_find) const noexcept {
        size_type rv = 0;
        for(const auto& x : *this)
            if(x == index_to_find) ++rv;
        return rv;
    }

    /** @brief Determines if *this is value equal to @p rhs.
     *
     *  *this and @p rhs are value equal if they contain the same number of
     *  indices and if the i-th dummy index of *this is value equal
     *  to the i-th dummy index of @p rhs for all i in the range [0, size()).
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const DummyIndices& rhs) const noexcept {
        return m_dummy_indices_ == rhs.m_dummy_indices_;
    }

    /** @brief Determines if *this is value equal to @p s.
     *
     *  This method is useful for comparing an already parsed DummyIndices
     *  object to a string literal. This method works by converting @p s into a
     *  DummyIndex object and then comparing the resulting DummyIndices object
     *  to *this. See operator==(const DummyIndices&) for more details on the
     *  comparison.
     *
     *  @param[in] s The string to compare to *this.
     *
     *  @return True if *this is value equal to @p s as a DummyIndices object
     *          and false otherwise.
     *
     *  @throw std::bad_alloc if parsing @p s encounters an allocation problem.
     *                        Strong throw guara
     *  @throw std::runtime_error if parsing @p s throws. Strong throw
     *                            guarantee.
     */
    bool operator==(const_reference s) const {
        return operator==(DummyIndices(s));
    }

    /** @brief Determines if *this is different from @p rhs.
     *
     *  Two DummyIndices objects are different if they are not value equal. See
     *  the description of operator==(const DummyIndices&) for the definition
     *  of value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return false if *this and @p rhs are value equal and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const DummyIndices& rhs) const noexcept {
        return !((*this) == rhs);
    }

    /** @brief Determines if *this is different than @p s.
     *
     *  This method is useful for comparing an already parsed DummyIndices
     *  object to a string literal. It works by converting @p s into a
     *  DummyIndices object and then comparing the resulting DummyIndices object
     *  to *this. See operator!=(const DummyIndices&) for more details on the
     *  comparison.
     *
     *  @param[in] s The string to compare to *this.
     *
     *  @return False if *this is value equal to @p s as a DummyIndices object
     *          and true otherwise.
     *
     *  @throw std::bad_alloc if parsing @p s encounters an allocation problem.
     *                        Strong throw guara
     *  @throw std::runtime_error if parsing @p s throws. Strong throw
     *                            guarantee.
     */
    bool operator!=(const_reference s) const { return !((*this) == s); }

    /** @brief Computes the DummyIndices object formed by concatenating *this
     *         with @p other.
     *
     *  This method will create a new DummyIndices object which contains
     *  `this->size()` plus `other.size()` indices. The first `this->size()`
     *  indices will be the indices of *this and the next `other.size()` indices
     *  will be the indices of @p other.
     *
     *  @note This is in general NOT the union of *this with @p other, in
     *        particular repeat indices may appear.
     *
     *  @param[in] other The indices to concatenate onto *this.
     *
     *  @return A new DummyIndices object formed by concatenating *this with
     *          @p other.
     *
     *  @throw std::bad_alloc if allocating the return fails. Strong throw
     *                        guarantee.
     */
    DummyIndices concatenation(const DummyIndices& other) const {
        DummyIndices rv(*this);
        for(const auto& x : other) rv.m_dummy_indices_.push_back(x);
        return rv;
    }

    /** @brief Returns the unique indices of *this which appear in @p other.
     *
     *  This method retruns a new DummyIndices object containing the set of
     *  indices in *this which also appear in @p other. The indices in the
     *  result are unique (i.e., if an index is repeated in *this it is only
     *  added to result once).
     *
     *  @param[in] other The object to compare *this to.
     *
     *  @return A DummyIndices object which contains the intersection of *this
     *          with @p other.
     *
     *  @throw std::bad_alloc if allocation of the return fails. Strong throw
     *                        guarantee.
     */
    DummyIndices intersection(const DummyIndices& other) const {
        DummyIndices rv;
        std::set<value_type> seen;
        for(const auto& x : *this) {
            if(seen.count(x)) continue;
            seen.insert(x);
            if(other.count(x)) rv.m_dummy_indices_.push_back(x);
        }
        return rv;
    }

    /** @brief Returns the  set difference of *this and @p other.
     *
     *  The set difference of *this with @p other is the set of indices which
     *  appear in *this, but not in @p other. This method will return the set
     *  (indices which appear more than once in *this will only appear once
     *  in the result) which results from the set difference of *this with
     *  @p other.
     *
     *  @param[in] other The set to remove from *this.
     *
     *  @return The set difference of *this and @p rhs.
     *
     *  @throw std::bad_alloc if there is a problem allocating the return.
     *                        Strong throw guarantee.
     */
    DummyIndices difference(const DummyIndices& other) const {
        DummyIndices rv;
        for(const auto& x : *this) {
            if(other.count(x)) continue;
            if(rv.count(x)) continue;
            rv.m_dummy_indices_.push_back(x);
        }
        return rv;
    }

protected:
    /// Lets the base class get at these implementations
    friend base_type;

    /// Implements mutable element retrieval by forwarding to m_dummy indices_
    reference at_(size_type i) { return m_dummy_indices_[i]; }

    /// Implements read-only element retrieval by forwarding to m_dummy indices_
    const_reference at_(size_type i) const { return m_dummy_indices_[i]; }

    /// Implements size by calling m_dummy indices_.size()
    size_type size_() const noexcept { return m_dummy_indices_.size(); }

private:
    /// Helper method for stripping spaces from the string @p input.
    static auto remove_spaces_(const_reference input) {
        value_type rv;
        for(const auto c : input)
            if(c != ' ') rv.push_back(c);
        return rv;
    }

    /// The split dummy indices
    split_string_type m_dummy_indices_;
};

template<typename StringType>
bool DummyIndices<StringType>::is_hadamard_product(
  const DummyIndices& lhs, const DummyIndices& rhs) const noexcept {
    if(has_repeated_indices()) return false;
    if(lhs.has_repeated_indices()) return false;
    if(rhs.has_repeated_indices()) return false;
    if(!is_permutation(lhs)) return false;
    if(!is_permutation(rhs)) return false;
    return true;
}

template<typename StringType>
bool DummyIndices<StringType>::is_contraction(
  const DummyIndices& lhs, const DummyIndices& rhs) const noexcept {
    if(has_repeated_indices()) return false;
    if(lhs.has_repeated_indices()) return false;
    if(rhs.has_repeated_indices()) return false;
    auto lhs_cap_rhs = lhs.intersection(rhs);
    if(lhs_cap_rhs.empty()) return false; // No common indices
    if(!intersection(lhs_cap_rhs).empty())
        return false; // Common index not summed
    return true;
}

} // namespace tensorwrapper::dsl