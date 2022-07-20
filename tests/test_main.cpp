#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <parallelzone/runtime.hpp>

int main(int argc, char* argv[]) {
    auto rt = parallelzone::Runtime(argc, argv);

    int res = Catch::Session().run(argc, argv);

    return res;
}
