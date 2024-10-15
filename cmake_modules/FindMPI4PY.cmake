# - FindMPI4PY
# Find mpi4py includes and library
# This module defines:
# MPI4PY_INCLUDE_DIR, where to find mpi4py.h, etc.
# MPI4PY_FOUND

# https://compacc.fnal.gov/projects/repositories/entry/synergia2/CMake/FindMPI4PY.cmake?rev=c147eafb60728606af4fe7b1b161a660df142e9a

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "python3")
endif(NOT PYTHON_EXECUTABLE)
execute_process(COMMAND "which" "${PYTHON_EXECUTABLE}")
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" "-c" "import mpi4py; print(mpi4py.get_include())"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
    RESULT_VARIABLE MPI4PY_COMMAND_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(MPI4PY_COMMAND_RESULT)
    message("jfa: mpi4py not found")
    set(MPI4PY_FOUND FALSE)
else(MPI4PY_COMMAND_RESULT)
    if(MPI4PY_INCLUDE_DIR MATCHES "Traceback")
        message("jfa: mpi4py matches traceback")
        ## Did not successfully include MPI4PY
        set(MPI4PY_FOUND FALSE)
    else(MPI4PY_INCLUDE_DIR MATCHES "Traceback")
        ## successful
        set(MPI4PY_FOUND TRUE)
        set(MPI4PY_INCLUDE_DIR
            ${MPI4PY_INCLUDE_DIR}
            CACHE STRING "mpi4py include path")
    endif(MPI4PY_INCLUDE_DIR MATCHES "Traceback")
endif(MPI4PY_COMMAND_RESULT)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPI4PY DEFAULT_MSG MPI4PY_INCLUDE_DIR)
