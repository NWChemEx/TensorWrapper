/*
 * Copyright 2025 NWChemEx-Project
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
#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/dsl/labeled.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::shape {
namespace {

/** @brief Recursively searches @p args for @p label.
 *
 *  @tparam Args the types of the labeled shapes to search.
 *
 *  To deal with an unknown number of labeled shapes we use recursion to loop
 *  over the list. Each invocation of `recurse_for_extent_` checks if @p label
 *  is found in @p shape. If it is, the extent is returned. If not, the
 *  parameter pack is unpacked into a new invocation of `recurse_for_extent_`
 *  and the process repeats.
 *
 *  @note This function short-circuits as soon as @p label is found and does not
 *        ensure that all shapes agree on the extend for @p label.
 *
 *  @param[in] label The label whose extent we are searching for.
 *  @param[in] shape The labeled shape to search at this recursion depth.
 *  @param[in] args  The remaining labeled shapes to search if @p label is not
 *                   found in @p shape.
 *
 *  @return The extent associated with @p label.
 *
 *  @throws std::runtime_error if @p label is not found in @p shape or any of
 *                             the objects in @p args. Strong throw guarantee.
 */
template<typename... Args>
auto recurse_for_extent_(const std::string& label,
                         dsl::Labeled<shape::ShapeBase> shape, Args&&... args) {
    auto idx = shape.labels().find(label);
    if(idx.empty()) {
        if constexpr(sizeof...(args) > 0) {
            return recurse_for_extent_(label, std::forward<Args>(args)...);
        } else {
            throw std::runtime_error("Label " + label +
                                     " not found in any provided shapes");
        }
    } else {
        return shape.object().as_smooth().extent(idx[0]);
    }
}

} // namespace

/** @brief Given a series of dummy indices and labeled shapes, works out the
 *         shape of the tensor described by the dummy indices.
 *
 *  @tparam StringType The string type used to represent the labels. Assumed to
 *                      be a type like std::string.
 *  @tparam Args The types of the labeled shapes provided.
 *
 *  This function wraps the process of working out the shape associated with a
 *  list of dummy indices. To do this, the function loops over each dummy
 *  index in @p labels and searches the labeled shapes in @p labeled_shapes for
 *  the dummy index. When the dummy index is found, the extent associated with
 *  the dummy index is recorded. If the dummy index is not found in any of the
 *  labeled shapes, an exception is thrown.
 *
 *  @param[in] labels The dummy indices describing the tensor whose shape is to
 *                    be determined.
 *  @param[in] labeled_shapes The labeled shapes to search for the dummy indices
 *                            in.
 *
 *  @return A Smooth shape describing the shape of the tensor with dummy indices
 *          @p labels.
 *
 *  @throw std::runtime_error if any of the labels in @p labels are not found
 *                            in @p labeled_shapes. Strong throw guarantee.
 */
template<typename StringType, typename... Args>
shape::Smooth shape_from_labels(const dsl::DummyIndices<StringType>& labels,
                                Args&&... labeled_shapes) {
    static_assert(sizeof...(Args) > 0,
                  "Must provide at least one labeled shape");

    std::vector<std::size_t> extents;
    for(const auto& label : labels) {
        extents.push_back(recurse_for_extent_(label, labeled_shapes...));
    }

    return shape::Smooth(extents.begin(), extents.end());
}

} // namespace tensorwrapper::shape
