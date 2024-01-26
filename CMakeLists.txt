# Copyright 2022 NWChemEx-Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" VERSION)
project(tensorwrapper VERSION "${VERSION}" LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
    nwx_cmake
    GIT_REPOSITORY https://github.com/NWChemEx/NWXCMake
)
FetchContent_MakeAvailable(nwx_cmake)
list(APPEND CMAKE_MODULE_PATH "${nwx_cmake_SOURCE_DIR}/cmake")

set(
    CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake"
    CACHE STRING "" FORCE
)

include(nwx_versions)
include(get_cmaize)
include(nwx_cxx_api_docs)
nwx_cxx_api_docs("${CMAKE_CURRENT_SOURCE_DIR}/src" 
                 "${CMAKE_CURRENT_SOURCE_DIR}/include")

### Options ###
option(BUILD_TESTING "Should we build the tests?" OFF)
option(BUILD_CPPYY_PYBINDINGS "Should we build Python3 bindings?" OFF)

cmaize_find_or_build_dependency(
    utilities
    URL github.com/NWChemEx/utilities
    VERSION ${NWX_UTILITIES_VERSION}
    BUILD_TARGET utilities
    FIND_TARGET nwx::utilities
    CMAKE_ARGS BUILD_TESTING=OFF
)

cmaize_find_or_build_dependency(
    parallelzone
    URL github.com/NWChemEx/ParallelZone
    VERSION ${NWX_PARALLELZONE_VERSION}
    BUILD_TARGET parallelzone
    FIND_TARGET nwx::parallelzone
    CMAKE_ARGS BUILD_TESTING=OFF
)

# N.B. If you update the TA version, you may also need to update 
#      the MADNESS version in the ParallelZone repo.
cmaize_find_or_build_dependency(
    tiledarray
    URL github.com/ValeevGroup/tiledarray
    VERSION ${NWX_TILEDARRAY_VERSION}
    BUILD_TARGET tiledarray
    FIND_TARGET tiledarray
    CMAKE_ARGS TA_BUILD_UNITTEST=OFF
               BUILD_TESTING=OFF 
               BLAS_THREAD_LAYER=sequential
               MADNESS_ENABLE_CEREAL=ON
)

find_package(Boost REQUIRED)

cmaize_add_library(
    tensorwrapper
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src/tensorwrapper"
    INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include/tensorwrapper"
    DEPENDS utilities parallelzone tiledarray Boost::boost
)

include(nwx_python_mods)
cppyy_make_python_package(PYTHONIZE MPI BLAS TILED PACKAGE tensorwrapper NAMESPACES tensorwrapper TA DEPPACKAGES parallelzone)

if("${BUILD_TESTING}")
    cmaize_find_or_build_dependency(
        Catch2
        URL github.com/catchorg/Catch2
        BUILD_TARGET Catch2
        FIND_TARGET Catch2::Catch2
        VERSION ${NWX_CATCH2_VERSION}
    )
    cmaize_add_tests(
        test_unit_tensorwrapper
        SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/tests/unit"
        INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src/tensorwrapper"
        DEPENDS Catch2 tensorwrapper
    )
    cmaize_add_tests(
        test_integration_tensorwrapper
        SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/tests/integration"
        INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src/tensorwrapper"
        DEPENDS Catch2 tensorwrapper
    )
endif()

cmaize_add_package(${PROJECT_NAME} NAMESPACE nwx::)
