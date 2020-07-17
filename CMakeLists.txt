cmake_minimum_required(VERSION 3.10)
if (${CMAKE_VERSION} VERSION_LESS 3.12)
	Project(Htool 
		VERSION 1.0
		LANGUAGES CXX)
else ()
	Project(Htool 
	VERSION 1.0
	DESCRIPTION "A header only c++ library that provides Hierarchical matrices."
	HOMEPAGE_URL "https://github.com/htool-ddm/htool"
	LANGUAGES CXX)
endif ()

# To force c++11
if (${CMAKE_VERSION} VERSION_LESS 3.1)
	add_compile_options(-std=c++11)
elseif (${CMAKE_VERSION} VERSION_LESS 3.6.3 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
	add_compile_options(-std=c++11)
else()
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

# To set default CMAKE_BUILD_TYPE
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Files to do find_package for some module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

# MPI
find_package (MPI REQUIRED)
message("-- MPI libraries found in " "${MPI_LIBRARIES}")
message("-- MPI include files found in " "${MPI_INCLUDE_PATH}")
separate_arguments(MPIEXEC_PREFLAGS) # to support multi flags

# OPENMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("-- OpenMP flags are " "${OpenMP_CXX_FLAGS}" )
    message("-- OpenMP linker flags are " "${OpenMP_EXE_LINKER_FLAGS}" )
else()
    message("-- OpenMP not found")
endif()

# BLAS
find_package(BLAS REQUIRED)
message("-- Found Blas implementation:" "${BLAS_LIBRARIES}")

# LAPACK
find_package(LAPACK)
message("-- Found Lapack:" "${LAPACK_LIBRARIES}")

# MPI4PY
find_package(MPI4PY REQUIRED)
message("-- Found MPI4PY:" "${MPI4PY_INCLUDE_DIR}")

add_subdirectory(lib/pybind11)

pybind11_add_module(Htool src/htool/main.cpp)
target_include_directories(Htool PRIVATE
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/htool/include>
                    $<INSTALL_INTERFACE:lib/htool/include>
                    lib/htool/include 
                    lib/hpddm/include
					${MPI_INCLUDE_PATH}
					${MKL_INC_DIR}
					${MPI4PY_INCLUDE_DIR})
target_link_libraries(Htool PRIVATE
					${MPI_LIBRARIES}
					${BLAS_LIBRARIES}
					${LAPACK_LIBRARIES}
					${ARPACK_LIBRARIES})

target_compile_options(Htool PRIVATE "-DPYTHON_INTERFACE")

# target_compile_features(Htool INTERFACE cxx_std_11)
