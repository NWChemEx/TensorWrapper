cmake_minimum_required(VERSION 3.14)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" VERSION)
project(tensorwrapper VERSION "${VERSION}" LANGUAGES CXX)

set(
    CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake"
    CACHE STRING "" FORCE
)

include(get_cpp)
include(nwx_cxx_api_docs)
nwx_cxx_api_docs("${CMAKE_CURRENT_SOURCE_DIR}/src" 
                 "${CMAKE_CURRENT_SOURCE_DIR}/include")

### Options ###
option(BUILD_TESTING "Should we build the tests?" OFF)
option(BUILD_PYBINDINGS "Should we build Python3 bindings?" ON)

cpp_find_or_build_dependency(
    utilities
    URL github.com/NWChemEx-Project/utilities
    PRIVATE TRUE
    BUILD_TARGET utilities
    FIND_TARGET nwx::utilities
    CMAKE_ARGS BUILD_TESTING=OFF
)

cpp_find_or_build_dependency(
    parallelzone
    URL github.com/NWChemEx-Project/ParallelZone
    PRIVATE TRUE
    BUILD_TARGET parallelzone
    FIND_TARGET nwx::parallelzone
    CMAKE_ARGS BUILD_TESTING=OFF
)

# N.B. If you update the TA version, you may also need to update 
#      the MADNESS version in the ParallelZone repo.
cpp_find_or_build_dependency(
    tiledarray
    URL github.com/ValeevGroup/tiledarray
    VERSION 1b7e2e3da438344a11c848724f3041b2d2b5c30b
    BUILD_TARGET tiledarray
    FIND_TARGET tiledarray
    CMAKE_ARGS TA_BUILD_UNITTEST=OFF
               BUILD_TESTING=OFF 
               BLAS_THREAD_LAYER=sequential
               MADNESS_ENABLE_CEREAL=ON
)

find_package(Boost REQUIRED COMPONENTS container)

cpp_add_library(
    tensorwrapper
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src/tensorwrapper"
    INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include/tensorwrapper"
    DEPENDS nwx::utilities nwx::parallelzone tiledarray Boost::container
)

include(nwx_python_mods)
cppyy_make_python_package(PYTHONIZE MPI BLAS TILED PACKAGE tensorwrapper NAMESPACES tensorwrapper TA DEPPACKAGES pluginplay)

if("${BUILD_TESTING}")
    cpp_find_or_build_dependency(
        Catch2
        URL github.com/catchorg/Catch2
        BUILD_TARGET Catch2
        VERSION v2.x
        FIND_TARGET Catch2::Catch2
        VERSION v2.x
    )
    cpp_add_tests(
        test_tensorwrapper
        SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/tests"
        INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/src/tensorwrapper"
        DEPENDS Catch2::Catch2 tensorwrapper
    )
endif()
