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
#include <tensorwrapper/dsl/parser.hpp>
#include <type_traits>
#include <utilities/dsl/dsl.hpp>
namespace tensorwrapper::dsl {

/** @brief Represents an object whose modes are assigned dummy indices.
 */
template<typename ObjectType, typename LabelType = std::string>
class Labeled : public utilities::dsl::BinaryOp<Labeled<ObjectType, LabelType>,
                                                ObjectType, LabelType> {
private:
    /// Type of *this
    using my_type = Labeled<ObjectType, LabelType>;

    /// Type *this inherits from
    using op_type = utilities::dsl::BinaryOp<my_type, ObjectType, LabelType>;

public:
    /// Reuse the base class's ctor
    using op_type::op_type;

    template<typename TermType>
    my_type& operator=(TermType&& other) {
        Parser<ObjectType, LabelType> p;
        *this = p.dispatch(std::move(*this), std::forward<TermType>(other));
        return *this;
    }
};

} // namespace tensorwrapper::dsl