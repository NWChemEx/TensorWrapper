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
