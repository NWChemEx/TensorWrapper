#include "../../../helpers.hpp"
#include "../../../inputs.hpp"
#include "../../../testing/eigen_buffers.hpp"
#include <tensorwrapper/dsl/executor/detail_/eigen_dispatcher.hpp>

using namespace tensorwrapper;

namespace {

template<typename... Args>
struct Checker {
    template<typename... Args2>
    Checker(Args2&&... args) : m_corr(std::forward<Args2>(args)...) {}

    auto run(Args... args) {
        auto inputs = std::tie(args...);
        REQUIRE(inputs == m_corr);
    }

    std::tuple<Args...> m_corr;
};

} // namespace

TEST_CASE("EigenDispatcher") {
    auto scalar = testing::eigen_scalar<double>();
    auto vector = testing::eigen_vector<double>();
    auto matrix = testing::eigen_matrix<double>();

    SECTION("EigenBuffer<double,0>, EigenBuffer<double, 1>") {
        Checker<testing::ebufferd0, testing::ebufferd1> c(scalar, vector);
    }
}