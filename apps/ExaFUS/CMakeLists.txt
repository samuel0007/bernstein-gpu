project(ExaFUS LANGUAGES CXX)
# Options
set(DIMENSION        3
    CACHE STRING "Spatial dimension (2, 3)")
set(LVARIANT "gll_warped"
    CACHE STRING "Basix LagrangeVariant (gll_warped, bernstein)")

set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX A6000
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS /home/sr2120/ascent/scripts/build_ascent/install/ascent-checkout)

add_executable(${PROJECT_NAME}
  main.cpp
)

# set(pyfile   ${CMAKE_CURRENT_SOURCE_DIR}/fus.py)
# set(cfile    ${CMAKE_CURRENT_BINARY_DIR}/fus.c)

# add_custom_command(
#   OUTPUT ${cfile}
#   COMMAND ${CMAKE_COMMAND} -E env
#           polynomial_degree=${polynomial_degree}
#           DIMENSION=${DIMENSION}
#           LVARIANT=${LVARIANT}
#           ffcx ${pyfile} ${SCALAR_TYPE}
#   DEPENDS ${pyfile}
#   COMMENT "Compiling forms.py -> forms.c"
#   VERBATIM
# )

link_common(${PROJECT_NAME} main.cpp)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    EXAFUS=1
    POLYNOMIAL_DEGREE=${polynomial_degree}
    SCALAR_TYPE_STR="${scalar_type}"
)

if(CMAKE_CUDA_COMPILER)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()