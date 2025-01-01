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

#include <stdexcept>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <tensorwrapper/dsl/pairwise_parser.hpp>
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper::dsl {
namespace {

using detail_::static_pointer_cast;

template<typename LHSType, typename RHSType>
Tensor tensor_assign(LHSType lhs, RHSType rhs) {
    auto playout =
      rhs.object().logical_layout().permute(rhs.labels(), lhs.labels());
    auto pdown   = static_pointer_cast<layout::Logical>(playout);
    auto pbuffer = rhs.object().buffer().permute(rhs.labels(), lhs.labels());
    return Tensor(std::move(pdown), std::move(pbuffer));
}

struct CallAddition {
    template<typename LHSType, typename RHSType>
    static decltype(auto) run(LHSType&& lhs, RHSType&& rhs) {
        return lhs.object().addition(lhs.labels(), rhs);
    }
};

template<typename FunctorType, typename ResultType, typename LHSType,
         typename RHSType>
Tensor tensor_binary(ResultType result, LHSType lhs, RHSType rhs) {
    Tensor buffer;
    if(result.object() == Tensor{}) {
        auto& llayout = lhs.object().logical_layout();
        auto lllayout = llayout(lhs.labels());
        auto& rlayout = rhs.object().logical_layout();
        auto lrlayout = rlayout(rhs.labels());
        auto playout  = FunctorType::run(lllayout, lrlayout);
        auto pdown    = static_pointer_cast<layout::Logical>(playout);

        auto lbuffer = lhs.object().buffer()(lhs.labels());
        auto rbuffer = rhs.object().buffer()(rhs.labels());
        auto pbuffer = FunctorType::run(lbuffer, rbuffer);

        Tensor(std::move(pdown), std::move(pbuffer)).swap(buffer);
    } else {
        throw std::runtime_error("Hints are not allowed yet!");
    }
    // No forwarding incase result appears multiple times in expression
    return tensor_assign(result, buffer(lhs.labels()));
}

} // namespace

#define TPARAMS template<typename ObjectType, typename LabelType>
#define PARSER PairwiseParser<ObjectType, LabelType>

TPARAMS
ObjectType PARSER::assign(const_labeled_type lhs, const_labeled_type rhs) {
    return tensor_assign(lhs, rhs);
}

TPARAMS
ObjectType PARSER::add(const_labeled_type result, const_labeled_type lhs,
                       const_labeled_type rhs) {
    return tensor_binary<CallAddition>(result, lhs, rhs);
}

#undef PARSER
#undef TPARAMS

template class PairwiseParser<Tensor, std::string>;

} // namespace tensorwrapper::dsl