cmake_minimum_required(VERSION 3.19)
enable_testing()

option(cpu "Build for CPUs"	off)
option(nvidia "Build for NVidia GPUs" on)
option(amd "Build for AMD GPUs"	off)

set(polynomial_degree 4 CACHE STRING "Polynomial degree")
set(scalar_type "float64" CACHE STRING "Scalar type")
message(STATUS "Polynomial degree: ${polynomial_degree}")
message(STATUS "Scalar type: ${scalar_type}")

if(NOT scalar_type STREQUAL "float32" AND NOT scalar_type STREQUAL "float64")
  message(FATAL_ERROR
    "Invalid scalar_type '${scalar_type}'. Valid values: float32, float64")
endif()

if(cpu AND nvidia OR cpu AND amd)
  message(FATAL_ERROR "You cannot select both a cpu build and a GPU (amd or nvidia) build. Only select one of the three options: cpu, nvidia, amd")
endif()
if(nvidia AND amd)
  message(FATAL_ERROR "You cannot select both amd and nvidia GPU builds. Only select one of the three options: cpu, nvidia, amd")
endif()

if(cpu)
  message(STATUS "Compiling for CPUs")
elseif(nvidia)
  message(STATUS "Compiling for NVIDIA GPUs")
elseif(amd)
  message(STATUS "Compiling for AMD GPUs")
else()
  message(FATAL_ERROR "No target selected for the build. Please select one of the three options: cpu, nvidia, amd")
endif()


if (scalar_type STREQUAL "float32")
  add_definitions(-DSCALAR_TYPE=float)
elseif(scalar_type STREQUAL "float64")
  add_definitions(-DSCALAR_TYPE=double)
else()
  message(FATAL_ERROR "Scalar type ${SCALAR_TYPE} not supported.")
endif()


set(PROJECT_NAME bernstein)
project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(amd)
  # set(CMAKE_HIP_ARCHITECTURES gfx90a gfx942 gfx1100)
  set(CMAKE_HIP_ARCHITECTURES gfx90a)
  enable_language(HIP)
elseif(nvidia)
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # set(CMAKE_CUDA_ARCHITECTURES 86 89 90a)  # RTX A4000
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX A4000
  # set(CMAKE_CUDA_ARCHITECTURES 80)  # A100
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

if (CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18) # See https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fclang-abi-compat=17")
endif()
# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -Ofast -DNDEBUG -munsafe-fp-atomics -march=native")

if (DEFINED ENV{CRAYPE_VERSION})
  message("Detected CrayPE - adding MPI flags")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -I/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include")
endif()

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -std=c++20 --extended-lambda --expt-relaxed-constexpr")

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

find_package(Boost 1.70 REQUIRED program_options)

include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
check_symbol_exists(PETSC_USE_COMPLEX petscsystypes.h PETSC_SCALAR_COMPLEX)
check_symbol_exists(PETSC_USE_REAL_DOUBLE petscsystypes.h PETSC_REAL_DOUBLE)

# Add target to compile UFL files
if(PETSC_SCALAR_COMPLEX EQUAL 1)
  if(PETSC_REAL_DOUBLE EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=complex128")
  else()
    set(SCALAR_TYPE "--scalar_type=complex64")
  endif()
else()
  if(PETSC_REAL_DOUBLE EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=float64")
  else()
    set(SCALAR_TYPE "--scalar_type=float32")
  endif()
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(bernsteinlib INTERFACE)
target_include_directories(bernsteinlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

function(link_common target)
  target_link_libraries(${target}
    PRIVATE
      dolfinx
      bernsteinlib
      Boost::program_options
  )

  if(amd)
    target_compile_definitions(${target} PRIVATE USE_HIP=1)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE HIP)
  elseif(nvidia)
    target_compile_definitions(${target} PRIVATE USE_CUDA=1)
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE CUDA)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CUDA)
  elseif(cpu)
    target_compile_definitions(${target} PRIVATE CPU_ONLY=1)
  endif()
endfunction()

function(add_example name is_test)
  set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/examples/${name})
  set(pyfile ${example_dir}/${name}.py)
  set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${name}.c)

  add_custom_command(
    OUTPUT ${cfile}
    COMMAND ${CMAKE_COMMAND} -E env polynomial_degree=${polynomial_degree} ffcx ${pyfile} ${SCALAR_TYPE}
    DEPENDS ${pyfile}
    COMMENT "Compiling ${name}.py"
    VERBATIM
  )

  add_executable(${name} ${example_dir}/main.cpp ${cfile})
  link_common(${name} ${example_dir}/main.cpp)

  target_compile_definitions(${name}
    PRIVATE
      DATA_DIR="${example_dir}"
      POLYNOMIAL_DEGREE=${polynomial_degree}
  )

  if(is_test)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name}
      PROPERTIES
        PASS_REGULAR_EXPRESSION "PASSED"
        FAIL_REGULAR_EXPRESSION "FAILED"
    )
  endif()
endfunction()

# Forward operators
add_example(mat_free_mass ON)
add_example(mat_free_mass3D ON)
add_example(mat_free_exterior_mass ON)
add_example(mat_free_exterior_mass3D ON)
add_example(mat_free_stiffness ON)
add_example(mat_free_stiffness3D ON)
add_example(mat_free_RHS ON)

# Solvers
add_example(mass_cg ON)
add_example(mass3D_cg ON)

# Acoustics Examples
add_example(planar_wave OFF)
add_example(planar_wave_triangles OFF)
add_example(planar_wave_triangles_gpu OFF)
add_example(planar_wave_tets_gpu OFF)
add_example(planar_wave_triangles_gpu_test OFF)
add_example(spherical_transducer_gpu OFF)
add_example(quad_gpu OFF)
add_example(hex_gpu OFF)


# Apps
add_subdirectory(apps/FreeFUS)
