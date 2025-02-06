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
#include <tensorwrapper/buffer/eigen.hpp>

namespace tensorwrapper::buffer {

/** @brief Relatively template-free API for Eigen tensor contraction.
 *
 *  Eigen's tensor library relies on a heavy amount of template meta-programming
 *  to implement contract. TensorWrapper strives to do things at runtime.
 *  Ultimately, to have it both ways we need to create contraction dispatch
 *  instantiations for every combination of template parameters that Eigen may
 *  end up seeing, that's what the functions in this header do.
 *
 */
template<typename FloatType, unsigned short Rank>
BufferBase::dsl_reference eigen_contraction(
  Eigen<FloatType, Rank>& result, BufferBase::label_type olabels,
  BufferBase::const_labeled_reference lhs,
  BufferBase::const_labeled_reference rhs);

} // namespace tensorwrapper::buffer