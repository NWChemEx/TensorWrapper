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

# Downloads common CMake modules used throughout NWChemEx
include(cmake/get_nwx_cmake.cmake)

#Sets the version to whatever git thinks it is
include(get_version_from_git)
get_version_from_git(tensorwrapper_version "${CMAKE_CURRENT_LIST_DIR}")
project(tensorwrapper VERSION "${tensorwrapper_version}" LANGUAGES CXX)

include(nwx_versions)
include(get_cmaize)

# Work out full paths to the project's include/src dirs
set(project_inc_dir "${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}")
set(project_src_dir "${CMAKE_CURRENT_LIST_DIR}/src/${PROJECT_NAME}")

include(nwx_cxx_api_docs)
nwx_cxx_api_docs("${project_inc_dir}" "${project_src_dir}")

### Options ###
cmaize_option_list(
    BUILD_TESTING OFF "Should we build the tests?"
    BUILD_PYBIND11_PYBINDINGS OFF "Should we build Python3 bindings?"
    ENABLE_EIGEN_SUPPORT ON "Should we enable Eigen support?"
)

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

find_package(Boost REQUIRED)

cmaize_find_or_build_optional_dependency(
    eigen
    ENABLE_EIGEN_SUPPORT
    URL https://www.gitlab.com/libeigen/eigen
    VERSION 3.4.0
    BUILD_TARGET eigen
    FIND_TARGET Eigen3::Eigen
)

cmaize_add_library(
    ${PROJECT_NAME}
    SOURCE_DIR "${project_src_dir}"
    INCLUDE_DIRS "${project_inc_dir}"
    DEPENDS utilities parallelzone Boost::boost eigen
)

if("${BUILD_TESTING}")
    set(cxx_test_dir "${CMAKE_CURRENT_LIST_DIR}/tests/cxx")
    set(cxx_unit_test_dir "${cxx_test_dir}/unit_tests/${PROJECT_NAME}")

    cmaize_find_or_build_dependency(
        Catch2
        URL github.com/catchorg/Catch2
        BUILD_TARGET Catch2
        FIND_TARGET Catch2::Catch2
        VERSION ${NWX_CATCH2_VERSION}
    )
    cmaize_add_tests(
        test_unit_${PROJECT_NAME}
        SOURCE_DIR "${cxx_unit_test_dir}"
        INCLUDE_DIRS "${project_src_dir}"
        DEPENDS Catch2 ${PROJECT_NAME}
    )
endif()

cmaize_add_package(${PROJECT_NAME} NAMESPACE nwx::)
