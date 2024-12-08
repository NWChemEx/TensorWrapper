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
#include "../../../allocator/detail_/eigen_buffer_unwrapper.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::dsl::executor::detail_ {

template<typename Functor>
class EigenDispatcher {
private:
    using rank_type = unsigned short;
    using unwrapper = allocator::detail_::EigenBufferUnwrapper;

    template<typename LHSType, typename RHSType>
    auto dispatch(LHSType&& lhs, RHSType&& rhs) {
        auto l = unwrapper::downcast(std::forward<LHSType>(lhs));
        auto r = unwrapper::downcast(std::forward<RHSType>(rhs));

        return std::visit(
          [](auto&& l_lambda) {
              return std::visit(
                [&l_lambda](auto&& r_lambda) {
                    Functor f;
                    return f.run(l_lambda, r_lambda);
                },
                r);
          },
          l);
    }
};

} // namespace tensorwrapper::dsl::executor::detail_