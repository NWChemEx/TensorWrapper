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
set(CMAKE_CXX_STANDARD 20)
project(tensorwrapper VERSION "${tensorwrapper_version}" LANGUAGES CXX)

include(nwx_versions)
include(get_cmaize)

# Work out full paths to the project's include/src dirs
set(project_inc_dir "${CMAKE_CURRENT_LIST_DIR}/include/${PROJECT_NAME}")
set(project_src_dir "${CMAKE_CURRENT_LIST_DIR}/src/${PROJECT_NAME}")

# Documentation
include(nwx_cxx_api_docs)
nwx_cxx_api_docs("${project_inc_dir}" "${project_src_dir}")

## Extensions ##
set(SOURCE_EXTS "cpp")
set(INCLUDE_EXTS "hpp")

### Options ###
cmaize_option_list(
    BUILD_TESTING OFF "Should we build the tests?"
    BUILD_PYBIND11_PYBINDINGS ON "Should we build Python3 bindings?"
    ENABLE_SIGMA OFF "Should we enable Sigma for uncertainty tracking?"
    ENABLE_CUTENSOR OFF "Should we enable cuTENSOR?"
)

if("${ENABLE_CUTENSOR}")
    if("${ENABLE_SIGMA}")
        set(MSG "Sigma is not compatible with cuTENSOR. Turning Sigma OFF.")
        message(WARNING ${MSG})
        set(ENABLE_SIGMA OFF)
    endif()
    enable_language(CUDA)
    set(SOURCE_EXTS ${SOURCE_EXTS} cu)
    set(INCLUDE_EXTS ${INCLUDE_EXTS} cuh hu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

### Dependendencies ###
include(get_utilities)

include(get_parallelzone)

find_package(Boost REQUIRED)

cmaize_find_or_build_dependency(
    eigen
    NAME Eigen3
    URL https://www.gitlab.com/libeigen/eigen
    VERSION 2e76277bd049f7bec36b0f908c69734a42c5234f
    BUILD_TARGET eigen
    FIND_TARGET Eigen3::Eigen
)

cmaize_find_or_build_optional_dependency(
    sigma
    ENABLE_SIGMA
    URL github.com/QCUncertainty/sigma
    VERSION main
    BUILD_TARGET sigma
    FIND_TARGET sigma::sigma
    CMAKE_ARGS BUILD_TESTING=OFF
               ENABLE_EIGEN_SUPPORT=ON
)

cmaize_find_or_build_dependency(
    WeaklyTypedFloat
    NAME WeaklyTypedFloat
    URL https://www.github.com/NWChemEx/weaklytypedfloat
    VERSION master
    BUILD_TARGET wtf
    FIND_TARGET nwx::wtf
)

set(DEPENDENCIES utilities parallelzone Boost::boost eigen sigma wtf)

if("${ENABLE_CUTENSOR}")
    include(cmake/FindcuTENSOR.cmake)
    list(APPEND DEPENDENCIES cuTENSOR::cuTENSOR)
endif()

cmaize_add_library(
    ${PROJECT_NAME}
    SOURCE_DIR "${project_src_dir}"
    SOURCE_EXTS "${SOURCE_EXTS}"
    INCLUDE_DIRS "${project_inc_dir}"
    INCLUDE_EXTS "${INCLUDE_EXTS}"
    DEPENDS "${DEPENDENCIES}"
)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

if("${ENABLE_CUTENSOR}")
    target_compile_definitions("${PROJECT_NAME}" PUBLIC ENABLE_CUTENSOR)
endif()

include(nwx_pybind11)
nwx_add_pybind11_module(
    ${PROJECT_NAME}
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/src/python"
    DEPENDS "${PROJECT_NAME}"
)


if("${BUILD_TESTING}")
    set(cxx_test_dir "${CMAKE_CURRENT_LIST_DIR}/tests/cxx")
    set(cxx_unit_test_dir "${cxx_test_dir}/unit_tests/${PROJECT_NAME}")
    set(
        cxx_acceptance_test_dir
        "${cxx_test_dir}/acceptance_tests/${PROJECT_NAME}"
    )
    set(
        cxx_performance_test_dir
        "${cxx_test_dir}/performance_tests/${PROJECT_NAME}"
    )
    set(python_test_dir "${CMAKE_CURRENT_LIST_DIR}/tests/python")
    set(python_unit_test_dir "${python_test_dir}/unit_tests")

    include(get_catch2)

    cmaize_add_tests(
        test_unit_${PROJECT_NAME}
        SOURCE_DIR "${cxx_unit_test_dir}"
        INCLUDE_DIRS "${project_src_dir}"
        DEPENDS Catch2 ${PROJECT_NAME}
    )

    cmaize_add_tests(
        test_performance_${PROJECT_NAME}
        SOURCE_DIR "${cxx_performance_test_dir}"
        INCLUDE_DIRS "${project_src_dir}"
        DEPENDS Catch2 ${PROJECT_NAME}
    )

    cmaize_add_tests(
        acceptance_test_${PROJECT_NAME}
        SOURCE_DIR "${cxx_acceptance_test_dir}"
        INCLUDE_DIRS "${project_src_dir}"
        DEPENDS Catch2 ${PROJECT_NAME}
    )


    nwx_add_pybind11_module(
        py_test_${PROJECT_NAME}
        INSTALL OFF
        SOURCE_DIR "${python_unit_test_dir}"
        INCLUDE_DIRS "${project_inc_dir}"
        DEPENDS ${PROJECT_NAME}
    )

    nwx_pybind11_tests(
        py_${PROJECT_NAME} ${python_unit_test_dir}/test_tensorwrapper.py
        SUBMODULES parallelzone pluginplay
        DEPENDS py_test_${PROJECT_NAME}
    )
endif()

cmaize_add_package(${PROJECT_NAME} NAMESPACE nwx::)
