#include "performance_testing.hpp"
#include <iostream>
#include <parallelzone/parallelzone.hpp>

using namespace tensorwrapper;

TEST_CASE("Contraction") {
    Tensor A{1.0, 2.0, 3.0};
    Tensor B{4.0, 5.0, 6.0};
    Tensor C;

    auto l = [&A, &B, &C]() {
        C("") = A("i") * B("i");
        return C;
    };

    parallelzone::hardware::CPU cpu;
    auto [rv, info] = cpu.profile_it(std::move(l));

    std::cout << "Time in ns: " << info.wall_time.count() << std::endl;
}