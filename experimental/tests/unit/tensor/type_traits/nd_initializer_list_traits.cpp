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

#include "tensorwrapper/tensor/type_traits/nd_initializer_list_traits.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("NDInitializerListTraits") {
    using scalar = double;
    using type0  = NDInitializerListTraits<scalar, 0ul>;
    using il0_t  = n_d_initializer_list_t<scalar, 0ul>;
    STATIC_REQUIRE(std::is_same_v<scalar, typename type0::type>);
    STATIC_REQUIRE(std::is_same_v<scalar, il0_t>);

    using vector_il = std::initializer_list<scalar>;
    using type1     = NDInitializerListTraits<scalar, 1ul>;
    using il1_t     = n_d_initializer_list_t<scalar, 1ul>;
    STATIC_REQUIRE(std::is_same_v<vector_il, typename type1::type>);
    STATIC_REQUIRE(std::is_same_v<vector_il, il1_t>);

    using matrix_il = std::initializer_list<vector_il>;
    using type2     = NDInitializerListTraits<scalar, 2ul>;
    using il2_t     = n_d_initializer_list_t<scalar, 2ul>;
    STATIC_REQUIRE(std::is_same_v<matrix_il, typename type2::type>);
    STATIC_REQUIRE(std::is_same_v<matrix_il, il2_t>);

    using tensor3_il = std::initializer_list<matrix_il>;
    using type3      = NDInitializerListTraits<scalar, 3ul>;
    using il3_t      = n_d_initializer_list_t<scalar, 3ul>;
    STATIC_REQUIRE(std::is_same_v<tensor3_il, typename type3::type>);
    STATIC_REQUIRE(std::is_same_v<tensor3_il, il3_t>);
}
