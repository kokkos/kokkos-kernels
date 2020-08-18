# Check whether the compiler defined the _Float16 type
# HAVE_KOKKOSKERNELS_FP16 is passed to C++ via KokkosKernels_config.h.in
INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE(_Float16 FP16 LANGUAGE CXX)
IF(HAVE_FP16)
  SET(HAVE_KOKKOSKERNELS_FP16 ${HAVE_FP16})
ENDIF()

# Check whether the cuda_fp16.h header exists to infer that the __half type exists
# HAVE_KOKKOSKERNELS_CUDA_FP16 is passed to C++ via KokkosKernels_config.h.in
INCLUDE(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(cuda_fp16.h HAVE_CUDA_FP16)
IF(HAVE_CUDA_FP16)
  SET(HAVE_CUDA_FP16 TRUE)
  SET(HAVE_KOKKOSKERNELS_CUDA_FP16 TRUE)
ELSE()
  SET(HAVE_CUDA_FP16 FALSE)
ENDIF()

IF(HAVE_KOKKOSKERNELS_FP16 AND HAVE_KOKKOSKERNELS_CUDA_FP16)
  MESSAGE(WARNING "'half' is set to 'device_fp16_t'. To use half precision on host, use 'host_fp16_t'.")
  MESSAGE(WARNING "Use 'float' and 'host_fp16_t' to cast on host.")
  MESSAGE(WARNING "Use '__half2float' and '__float2half' functions to cast on device.")
ENDIF()

# HAVE_KOKKOSKERNELS_HALFMATH is passed to C++ via KokkosKernels_config.h.in
IF(HAVE_KOKKOSKERNELS_FP16 OR HAVE_KOKKOSKERNELS_CUDA_FP16)
  SET(HAVE_KOKKOSKERNELS_HALFMATH TRUE)
ENDIF()
