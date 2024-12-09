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

#include "../../../helpers.hpp"
#include "../../../inputs.hpp"
#include "../../../testing/eigen_buffers.hpp"
#include <tensorwrapper/dsl/executor/detail_/eigen_assign.hpp>
#include <tensorwrapper/dsl/executor/detail_/eigen_dispatcher.hpp>
using namespace tensorwrapper;
using namespace testing;

namespace {

template<typename... Args>
struct Checker {
    template<typename... Args2>
    Checker(Args2&&... args) : m_corr(std::forward<Args2>(args)...) {}

    template<typename... Args2>
    auto run(Args2&&... args) {
        using clean_t = std::tuple<std::decay_t<Args2>...>;
        if constexpr(std::is_same_v<clean_t, std::tuple<Args...>>) {
            auto inputs = std::tie(std::forward<Args2>(args)...);
            REQUIRE(inputs == m_corr);
        } else {
            throw std::runtime_error("Unsupported tuple of buffers");
        }
    }

    std::tuple<Args...> m_corr;
};

} // namespace

TEST_CASE("EigenDispatcher") {
    auto scalar = eigen_scalar<double>();
    auto vector = eigen_vector<double>();
    auto matrix = eigen_matrix<double>();

    SECTION("Eigen<double,0>") {
        Checker<ebufferd0> c(scalar);
        dsl::executor::detail_::EigenDispatcher d(std::move(c));
        d.dispatch(scalar);
    }

    SECTION("Eigen<double,0>, Eigen<double,1>") {
        Checker<ebufferd0, ebufferd1> c(scalar, vector);
        dsl::executor::detail_::EigenDispatcher d(std::move(c));
        d.dispatch(scalar, vector);
    }

    SECTION("Eigen<double,1>, Eigen<double,2>, Eigen<double,0>") {
        Checker<ebufferd1, ebufferd2, ebufferd0> c(vector, matrix, scalar);
        dsl::executor::detail_::EigenDispatcher d(std::move(c));
        d.dispatch(vector, matrix, scalar);
    }

    SECTION("assignment") {
        dsl::executor::detail_::EigenAssign c;
        dsl::executor::detail_::EigenDispatcher d(std::move(c));
        buffer::Eigen<double, 0> scalar2;
        auto pscalar2 = &(d.dispatch(scalar2, scalar));
        REQUIRE(pscalar2 == &scalar2);
        REQUIRE(scalar2.value()() == 42.0);
    }
}