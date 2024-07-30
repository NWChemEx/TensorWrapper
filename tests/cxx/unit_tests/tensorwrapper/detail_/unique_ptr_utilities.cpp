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

#include <catch2/catch.hpp>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <vector>

using namespace tensorwrapper::detail_;

namespace {
struct BaseClass {
    virtual ~BaseClass() = default;
};
struct DerivedClass : public BaseClass {
    int x = 2;
};
} // namespace

TEST_CASE("static_pointer_cast") {
    auto pderived               = std::make_unique<DerivedClass>();
    DerivedClass* pderived_corr = pderived.get();

    std::unique_ptr<BaseClass> pbase(std::move(pderived));

    SECTION("Good cast") {
        auto pdowncast = static_pointer_cast<DerivedClass>(pbase);

        REQUIRE(pbase.get() == nullptr);
        REQUIRE(pdowncast.get() == pderived_corr);
    }
    // This next line shouldn't compile. Uncomment to test.
    // SECTION("Bad cast") { static_pointer_cast<std::vector<double>>(pbase); }
}

TEST_CASE("dynamic_pointer_cast") {
    auto pderived               = std::make_unique<DerivedClass>();
    DerivedClass* pderived_corr = pderived.get();

    std::unique_ptr<BaseClass> pbase(std::move(pderived));

    SECTION("Good cast") {
        auto pdowncast = dynamic_pointer_cast<DerivedClass>(pbase);

        REQUIRE(pbase.get() == nullptr);
        REQUIRE(pdowncast.get() == pderived_corr);
    }

    SECTION("Bad cast") {
        BaseClass* pbase_corr = pbase.get();
        auto pbadcast = dynamic_pointer_cast<std::vector<double>>(pbase);

        REQUIRE(pbase.get() == pbase_corr);
        REQUIRE(pbadcast.get() == nullptr);
    }
}
