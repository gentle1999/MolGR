cmake_minimum_required(VERSION 3.15)
project(molgr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

# ==============================================================================
# 1. 基础依赖查找
# ==============================================================================
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# >>> 修复核心: 动态设置 pybind11 路径 <<<
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c 
    "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${_pybind11_dir}")
message(STATUS "Pybind11 CMake dir: ${pybind11_DIR}")

find_package(pybind11 REQUIRED)

# ==============================================================================
# 2. 定位 OpenBabel (根据 openbabel-wheel 的结构适配)
# ==============================================================================
message(STATUS ">>> CMake is using Python: ${Python_EXECUTABLE}")
# 2.1 使用 Python 动态获取 openbabel 包的安装根目录
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c 
    "import openbabel; import os; print(os.path.dirname(openbabel.__file__), end='')"
    OUTPUT_VARIABLE OB_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "OpenBabel package root: ${OB_ROOT}")

# 2.2 设置头文件路径
set(OB_INCLUDE_DIR "${OB_ROOT}/include/openbabel3")


# 2.3 设置库文件路径
if(WIN32)
    set(OB_LIBRARY_NAMES openbabel-3 openbabel)
    set(OB_LIBRARY_PATHS "${OB_ROOT}/bin" "${OB_ROOT}/lib" "${OB_ROOT}")
else()
    set(OB_LIBRARY_NAMES openbabel)
    set(OB_LIBRARY_PATHS "${OB_ROOT}/lib" "${OB_ROOT}")
endif()

find_library(OB_LIBRARY
    NAMES ${OB_LIBRARY_NAMES}
    PATHS ${OB_LIBRARY_PATHS}
    NO_DEFAULT_PATH
)

message(STATUS "OpenBabel library: ${OB_LIBRARY}")

if(NOT OB_LIBRARY)
    message(FATAL_ERROR "Could not find the Open Babel library under ${OB_ROOT}")
endif()

# ==============================================================================
# 3. 构建目标
# ==============================================================================

add_library(molgr STATIC
    src/cpp/src/process_guard.cpp
    src/cpp/src/state.cpp
    src/cpp/src/utils/consts.cpp
    src/cpp/src/utils/utils.cpp
    src/cpp/src/utils/conversions.cpp
    src/cpp/src/utils/stereo.cpp
    src/cpp/src/utils/equivalence.cpp
    src/cpp/src/utils/equivalence_stereo.cpp
    src/cpp/src/utils/xyz.cpp
    src/cpp/src/utils/force_field.cpp
    src/cpp/src/utils/metal_radical_inference.cpp
    src/cpp/src/vendor/forcefielduff.cpp
    src/cpp/src/vendor/openbabel_conversion.cpp
    src/cpp/src/vendor/openbabel_copy.cpp
    src/cpp/src/vendor/openbabel_threading.cpp
    src/cpp/src/utils/metals/preparation.cpp
    src/cpp/src/utils/metals/scoring.cpp
    src/cpp/src/utils/metals/search.cpp
    src/cpp/src/utils/no_metals/neighbor_radicals.cpp
    src/cpp/src/utils/no_metals/preparation.cpp
    src/cpp/src/utils/no_metals/recovery.cpp
    src/cpp/src/utils/no_metals/resonance.cpp
    src/cpp/src/utils/no_metals/selection.cpp
    src/cpp/src/utils/organic_topology.cpp
    src/cpp/src/utils/perf.cpp
    src/cpp/src/utils/resonance.cpp
    src/cpp/src/utils/smarts.cpp
    src/cpp/src/utils/scoring.cpp
    src/cpp/src/utils/logger.cpp
    src/cpp/src/pipeline/reconstruct_with_metals.cpp
    src/cpp/src/pipeline/reconstruct_batch.cpp
    src/cpp/src/pipeline/reconstruct_without_metals.cpp
    src/cpp/src/pipeline/resonance.cpp
    src/cpp/src/stages/preprocess.cpp
    src/cpp/src/stages/fresh.cpp
    src/cpp/src/stages/eliminate.cpp
    src/cpp/src/stages/clean.cpp
    src/cpp/src/stages/break_bond.cpp
)

set_target_properties(molgr PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1950)
    target_compile_options(molgr PUBLIC
        "/FImolgr/compat/msvc_openbabel_stl.h"
    )
endif()

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_definitions(molgr PUBLIC _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
endif()

target_include_directories(molgr PUBLIC
    src/cpp/include
    ${OB_INCLUDE_DIR}
)

target_link_libraries(molgr PUBLIC ${OB_LIBRARY})

# ==============================================================================
# 4. 定义 Pybind11 模块 (_core)
# ==============================================================================
pybind11_add_module(_core 
    src/bindings/module.cpp 
    src/bindings/bind_utils.cpp
    src/bindings/bind_equivalence.cpp
    src/bindings/bind_pipeline.cpp
    src/bindings/bind_stages.cpp
    src/cpp/src/config.cpp
)

target_link_libraries(_core PRIVATE molgr)

if(APPLE)
    set_target_properties(_core PROPERTIES
        MACOSX_RPATH ON
        BUILD_WITH_INSTALL_RPATH ON
        BUILD_RPATH "${OB_ROOT}/lib;${OB_ROOT}"
        INSTALL_RPATH "@loader_path/../openbabel/lib;@loader_path/../openbabel"
    )
elseif(UNIX)
    set_target_properties(_core PROPERTIES BUILD_RPATH "${OB_ROOT}/lib:${OB_ROOT}")
    set_target_properties(_core PROPERTIES INSTALL_RPATH "$ORIGIN/../openbabel/lib:$ORIGIN/../openbabel")
endif()

install(TARGETS _core DESTINATION molgr)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/molgr/data/UFF.prm" DESTINATION molgr/data)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/molgr/data/RDKIT_STEREO_LICENSE.txt" DESTINATION molgr/data)
