cmake_minimum_required(VERSION 3.10)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    project(
        Htool
        VERSION 1.0.0
        LANGUAGES CXX)
else()
    project(
        Htool
        VERSION 1.0.0
        DESCRIPTION "Python interface of Htool-DDM, a header only c++ library that provides Hierarchical matrices."
        HOMEPAGE_URL "https://github.com/htool-ddm/htool_python"
        LANGUAGES CXX)
endif()

# To force c++17
if(${CMAKE_VERSION} VERSION_LESS 3.1)
    add_compile_options(-std=c++17)
elseif(${CMAKE_VERSION} VERSION_LESS 3.6.3 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    add_compile_options(-std=c++17)
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# To set default CMAKE_BUILD_TYPE
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "${default_build_type}"
        CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Formatting
include(cmake/formatting.cmake)
file(GLOB_RECURSE ALL_CODE_FILES ${PROJECT_SOURCE_DIR}/src/*.[h]pp)

clang_format(format ${ALL_CODE_FILES})
cmake_format(cmake_format ${PROJECT_SOURCE_DIR}/CMakeLists.txt)

# Files to do find_package for some module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

# MPI
find_package(MPI)
if(MPI_FOUND)
    message("-- MPI libraries found in " "${MPI_LIBRARIES}")
    message("-- MPI include files found in " "${MPI_INCLUDE_PATH}")
    separate_arguments(MPIEXEC_PREFLAGS) # to support multi flags
endif()
# OPENMP
find_package(OpenMP)

# BLAS
find_package(BLAS REQUIRED)
message("-- Blas implementation:" "${BLAS_LIBRARIES}")

# LAPACK
find_package(LAPACK)
message("-- Lapack implementation:" "${LAPACK_LIBRARIES}")

# MPI4PY
find_package(
    Python
    COMPONENTS Interpreter Development
    REQUIRED)
message("-- Python executable:" "${Python_EXECUTABLE}")

# MPI4PY
if(MPI_FOUND)
    find_package(MPI4PY REQUIRED)
    message("-- MPI4PY include dir:" "${MPI4PY_INCLUDE_DIR}")
endif()

add_subdirectory(lib/pybind11)

pybind11_add_module(Htool src/htool/main.cpp)
target_include_directories(Htool PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/htool/include> $<INSTALL_INTERFACE:lib/htool/include> lib/htool/include lib/hpddm/include ${MKL_INC_DIR})
if(MPI_FOUND)
    target_link_libraries(Htool PRIVATE MPI::MPI_CXX)
    target_include_directories(Htool PRIVATE ${MPI_INCLUDE_PATH} ${MPI4PY_INCLUDE_DIR})
    target_compile_definitions(Htool PRIVATE HAVE_MPI)
endif()
target_link_libraries(Htool PRIVATE ${MPI_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${ARPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

if("${BLA_VENDOR}" STREQUAL "Intel10_32"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64lp"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64lp_seq"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64ilp"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64ilp_seq"
   OR "${BLA_VENDOR}" STREQUAL "Intel10_64_dyn")
    target_compile_definitions(Htool PRIVATE "-DHPDDM_MKL -DHTOOL_MKL")
endif()

target_compile_definitions(Htool PRIVATE "-DHTOOL_WITH_PYTHON_INTERFACE" "-DHTOOL_WITH_HPDDM")

if(CODE_COVERAGE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(STATUS "Code coverage enabled with GNU.")
        target_compile_options(Htool PRIVATE -fprofile-arcs -ftest-coverage -fno-elide-constructors -fno-default-inline -O0)
        target_link_options(
            Htool
            PRIVATE
            -fprofile-arcs
            -ftest-coverage
            -fno-elide-constructors
            -fno-default-inline
            -O0)
        target_link_libraries(Htool PRIVATE gcov)
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "Code coverage enabled with LLVM.")
        target_compile_options(Htool PRIVATE -fprofile-instr-generate -fcoverage-mapping -O0)
        target_link_options(Htool PRIVATE -fprofile-instr-generate -fcoverage-mapping -O0)
    else()
        message(STATUS "Code coverage not available.")
    endif()
endif()
