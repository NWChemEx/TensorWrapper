# Copyright 2026 NWChemEx-Project
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

# Downloads common CMake modules used throughout NWChemEx. Included before
# project() so nwx_set_version() is available to compute the version.
include(cmake/get_nwx_cmake.cmake)

# Version from scikit-build-core (wheels/sdists) or the latest git tag.
include(nwx_set_version)
nwx_set_version(tw_version "${CMAKE_CURRENT_LIST_DIR}")
project(tensorwrapper VERSION "${tw_version}" LANGUAGES CXX)

include(disable_in_source_builds)
include(set_default_nwx_options)
option(ENABLE_CUTENSOR         "Enable cuTENSOR GPU backend"           OFF)

# Documentation target
include(nwx_cxx_api_docs)
nwx_cxx_api_docs("cxx/include" "cxx/src")

# Dependencies
include(get_dependencies)
get_dependencies(utilities parallelzone wtf eigen boost)

# TODO: ENABLE_CUTENSOR handling

# Build the library
include(nwx_library)
nwx_library(${PROJECT_NAME} "cxx/include" "cxx/src"
    utilities parallelzone wtf
    PUBLIC eigen boost)

# Generated config headers (e.g. from configure_file) live in binary dir
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

# Python module
include(nwx_python_module)
nwx_python_module(${PROJECT_NAME} "cxx/src")

# C++ tests
include(catch2_tests_from_dir)
catch2_tests_from_dir(
    "unit_test_${PROJECT_NAME}"
    "tests/cxx/unit_tests/${PROJECT_NAME}"
    ${PROJECT_NAME}
    PRIVATE_INCLUDES cxx/src
)
catch2_tests_from_dir(
    "test_acceptance_${PROJECT_NAME}"
    "tests/cxx/acceptance_tests/${PROJECT_NAME}"
    ${PROJECT_NAME}
    PRIVATE_INCLUDES cxx/src
)
catch2_tests_from_dir(
    "test_performance_${PROJECT_NAME}"
    "tests/cxx/performance_tests/${PROJECT_NAME}"
    ${PROJECT_NAME}
    PRIVATE_INCLUDES cxx/src
)

# py_test_tensorwrapper: a pybind11 module exposing C++ test fixtures to
# tests/python/unit_tests/*.py (e.g. tensor/test_tensor.py imports
# py_test_tensorwrapper.testing). Not part of the installed package --
# test-only, built in place alongside the other test binaries.
if(BUILD_PYBIND11_BINDINGS AND BUILD_TESTING)
    nwx_python_module(py_test_tensorwrapper "tests/python/unit_tests" NO_INSTALL)
endif()
