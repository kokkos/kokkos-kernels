kokkoskernels_add_option("INST_DOUBLE" ${KOKKOSKERNELS_ADD_DEFAULT_ETI} BOOL
  "Whether to pre instantiate kernels for the scalar type double.  This option is KokkosKernels_INST_DOUBLE=ON by default.  Disabling this may increase build times. Default: ON")

if(Trilinos_ENABLE_FLOAT)
  set(KOKKOSKERNELS_INST_FLOAT_DEFAULT ON)
else()
  set(KOKKOSKERNELS_INST_FLOAT_DEFAULT OFF)
endif()

kokkoskernels_add_option("INST_FLOAT" ${KOKKOSKERNELS_INST_FLOAT_DEFAULT} BOOL
  "Whether to pre instantiate kernels for the scalar type float.  Disabling this may increase build times. Default: OFF or unless enabled during a Trilinos build with Trilinos_ENABLE_FLOAT.")

kokkoskernels_add_option("INST_HALF" OFF BOOL
  "Whether to pre instantiate kernels for the scalar type Kokkos::Experimental::half_t.  Disabling this may increase build times. Default: OFF")

# Detect whether Kokkos defines KOKKOS_HAS_NATIVE_HALF_TYPE, i.e. whether
# half_t is a natively supported 16-bit type rather than an alias for float.
# This test uses the proper compiler launcher from Kokkos.
function(kokkoskernels_check_native_half RESULT_VAR)
  include(CheckCXXSymbolExists)
  get_property(_launcher GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
  if(_launcher)
    separate_arguments(CMAKE_CXX_COMPILER_LAUNCHER UNIX_COMMAND "${_launcher}")
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_CXX_COMPILER_LAUNCHER)
  endif()
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  set(CMAKE_REQUIRED_LIBRARIES Kokkos::kokkos)
  check_cxx_symbol_exists(KOKKOS_HAS_NATIVE_HALF_TYPE "Kokkos_Macros.hpp" ${RESULT_VAR})
endfunction()
kokkoskernels_check_native_half(KOKKOS_CORE_HAS_HALF)

if (KOKKOSKERNELS_INST_HALF AND KOKKOS_CORE_HAS_HALF)
set(REAL_FLOATS              HALF FLOAT DOUBLE)
set(FLOATS                   HALF FLOAT DOUBLE COMPLEX_FLOAT COMPLEX_DOUBLE)
else()
set(REAL_FLOATS              FLOAT DOUBLE)
set(FLOATS                   FLOAT DOUBLE COMPLEX_FLOAT COMPLEX_DOUBLE)
endif()
set(DOUBLE_CPP_TYPE          "double")
set(FLOAT_CPP_TYPE           "float")
set(HALF_CPP_TYPE            "Kokkos::Experimental::half_t")
set(COMPLEX_FLOAT_CPP_TYPE   "Kokkos::complex<float>")
set(COMPLEX_DOUBLE_CPP_TYPE  "Kokkos::complex<double>")

#Just leave the Trilinos logic here alone
#Outside of Trilinos, let the defaults be always OFF
if(KOKKOSKERNELS_INST_DOUBLE AND Trilinos_ENABLE_COMPLEX_DOUBLE)
  set(KOKKOSKERNELS_INST_COMPLEX_DOUBLE_DEFAULT ON)
else()
  set(KOKKOSKERNELS_INST_COMPLEX_DOUBLE_DEFAULT OFF)
endif()
if(KOKKOSKERNELS_INST_FLOAT AND Trilinos_ENABLE_COMPLEX_FLOAT)
  set(KOKKOSKERNELS_INST_COMPLEX_FLOAT_DEFAULT ON)
else()
  set(KOKKOSKERNELS_INST_COMPLEX_FLOAT_DEFAULT OFF)
endif()

kokkoskernels_add_option("INST_COMPLEX_DOUBLE" ${KOKKOSKERNELS_INST_COMPLEX_DOUBLE_DEFAULT} BOOL
  "Whether to pre instantiate kernels for the scalar type complex<double>.  Disabling this may increase build times. Default: OFF or unless enabled during a Trilinos build with Trilinos_ENABLE_COMPLEX_DOUBLE.")

kokkoskernels_add_option("INST_COMPLEX_FLOAT" ${KOKKOSKERNELS_INST_COMPLEX_FLOAT_DEFAULT} BOOL
  "Whether to pre instantiate kernels for the scalar type complex<float>.  Disabling this may increase build times. Default: OFF or unless enabled during a Trilinos build with Trilinos_ENABLE_COMPLEX_FLOAT.")

if(KOKKOSKERNELS_INST_DOUBLE)
  list(APPEND SCALAR_LIST "double")
endif()

if(KOKKOSKERNELS_INST_FLOAT)
  list(APPEND SCALAR_LIST "float")
endif()

if (KOKKOSKERNELS_INST_HALF)
 list(APPEND SCALAR_LIST "Kokkos::Experimental::half_t")
endif()

if(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)
  list(APPEND SCALAR_LIST "complex<double>")
endif()

if(KOKKOSKERNELS_INST_COMPLEX_FLOAT)
  list(APPEND SCALAR_LIST "complex<float>")
endif()
