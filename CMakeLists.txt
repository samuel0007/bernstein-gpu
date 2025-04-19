cmake_minimum_required(VERSION 3.19)

option(cpu "Build for CPUs"	off)
option(nvidia "Build for NVidia GPUs" on)
option(amd "Build for AMD GPUs"	off)

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


set(PROJECT_NAME bernstein)
project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(amd)
  set(CMAKE_HIP_ARCHITECTURES gfx90a gfx942 gfx1100)
  enable_language(HIP)
elseif(nvidia)
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # set(CMAKE_CUDA_ARCHITECTURES 86 89 90a)  # RTX A4000
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX A4000
  set(CMAKE_CUDA_STANDARD 20)
  enable_language(CUDA)
endif()

if (CMAKE_HIP_COMPILER_VERSION GREATER_EQUAL 18) # See https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.2.1/LLVM/clang/html/ReleaseNotes.html#c-specific-potentially-breaking-changes
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fclang-abi-compat=17")
endif()
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -std=c++20 -Wall -Ofast -DNDEBUG -munsafe-fp-atomics")

if (DEFINED ENV{CRAYPE_VERSION})
   message("Detected CrayPE - adding MPI flags")
   set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Ofast -march=native -I/opt/cray/pe/mpich/8.1.27/ofi/gnu/9.1/include")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -march=native -std=c++20 -O3 -DNDEBUG --extended-lambda --expt-relaxed-constexpr -lineinfo")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -std=c++20 --extended-lambda --expt-relaxed-constexpr")

set(VERBOSE)

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

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

function(add_example name)
  set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/examples/${name})
  set(pyfile ${example_dir}/${name}.py)
  set(cfile ${CMAKE_CURRENT_BINARY_DIR}/${name}.c)


  add_custom_command(
    OUTPUT ${cfile}
    COMMAND ffcx ${pyfile} ${SCALAR_TYPE}
    DEPENDS ${pyfile}
    COMMENT "Compiling ${name}.py"
    VERBATIM
  )

  add_executable(${name} ${example_dir}/main.cpp ${cfile})
  target_link_libraries(${name} PRIVATE dolfinx bernsteinlib)

  target_compile_definitions(${name}
    PRIVATE
      DATA_DIR="${example_dir}"
  )

  if(amd)
    target_compile_definitions(${name} PRIVATE USE_HIP=1)
    set_source_files_properties(${example_dir}/main.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE HIP)
  elseif(nvidia)
    target_compile_definitions(${name} PRIVATE USE_CUDA=1)
    set_source_files_properties(${example_dir}/main.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CUDA)
  elseif(cpu)
    target_compile_definitions(${name} PRIVATE CPU_ONLY=1)
  endif()
endfunction()

add_example(mat_free_mass)
add_example(mass_cg)
add_example(planar_wave)
add_example(planar_wave_triangles)


# Do not throw error for 'multi-line comments' (these are typical in rst which
# includes LaTeX)
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-Wno-comment" HAVE_NO_MULTLINE)
# set_source_files_properties(
#   main.cpp
#   PROPERTIES
#     COMPILE_FLAGS
#     "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>"
# )