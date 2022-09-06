/*
 * Copyright 2022 NWChemEx-Project
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
#include <numeric>

namespace tensorwrapper::ta_helpers::detail_ {

/// Generic functor satisfying TiledArray's Reducer API
template<typename TensorType, typename AddOp, typename TimesOp>
class Reducer {
    using tile_type    = typename TensorType::value_type;
    using element_type = typename tile_type::value_type;

public:
    // Required typedefs
    using result_type =
      std::invoke_result_t<TimesOp, element_type, element_type>;
    using first_argument_type  = tile_type;
    using second_argument_type = tile_type;

    using result_ref       = result_type&;
    using const_result_ref = const result_type&;
    using const_arg1_ref   = const first_argument_type&;
    using const_arg2_ref   = const second_argument_type&;

    Reducer(AddOp add_op, TimesOp times_op, result_type init) :
      m_add_op_(std::move(add_op)),
      m_times_op_(std::move(times_op)),
      m_init_(std::move(init)) {}

    result_type operator()() const { return m_init_; }

    const_result_ref operator()(const_result_ref r) const { return r; }

    void operator()(result_ref result, const_result_ref arg) const {
        result = m_add_op_(result, arg);
    }

    void operator()(result_ref result, const_arg1_ref first,
                    const_arg2_ref second) const {
        result = std::inner_product(first.begin(), first.end(), second.begin(),
                                    result, m_add_op_, m_times_op_);
    }

private:
    AddOp m_add_op_;
    TimesOp m_times_op_;
    result_type m_init_;
};

} // namespace tensorwrapper::ta_helpers::detail_
