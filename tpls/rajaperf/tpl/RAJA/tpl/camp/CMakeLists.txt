cmake_minimum_required (VERSION 3.9)

project (camp
  LANGUAGES CXX
  VERSION 0.1.0)

option(ENABLE_CUDA "whether to build with cuda" Off)
option(ENABLE_CLANG_CUDA "use clang cuda support rather than nvcc" Off)
option(ENABLE_HIP "whether to build with hip" Off)
option(ENABLE_TARGET_OPENMP "whether to build with OpenMP offload" Off)
option(ENABLE_TESTS "whether to configure and build tests" On)

if(ENABLE_CLANG_CUDA)
  if(NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
    message(ERROR "clang cuda can only be used with clang")
  endif()
endif()

if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  if(NOT ENABLE_CLANG_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
  else()
    if(NOT CUDA_ARCH)
      set(CUDA_ARCH "sm_35")
    endif()
    set (_cuda_compile_flags -x cuda --cuda-gpu-arch=${CUDA_ARCH} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR})
    add_library(camp_cuda INTERFACE)
    target_compile_options(camp_cuda INTERFACE ${_cuda_compile_flags})
    target_link_libraries(camp_cuda INTERFACE ${CUDA_LIBRARIES})
    target_include_directories(camp_cuda INTERFACE ${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA LIBS: ${CUDA_LIBRARIES}")
  endif(NOT ENABLE_CLANG_CUDA)
endif()

add_library (camp INTERFACE)
target_include_directories (camp INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )
set_target_properties (camp PROPERTIES
  INTERFACE_LIB_VERSION $camp_VERSION
  INTERFACE_COMPILE_FEATURES cxx_std_11)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  VERSION 0.1.0
  COMPATIBILITY AnyNewerVersion
  )

install(TARGETS camp
  EXPORT campTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION include COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
  )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/campConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/campConfig.cmake"
  INSTALL_DESTINATION
  lib/cmake/camp
  )

install(EXPORT campTargets
  DESTINATION lib/cmake/camp)
install(FILES
  "${PROJECT_BINARY_DIR}/campConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/campConfig.cmake"
  DESTINATION
  lib/cmake/camp)
install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/include/
  DESTINATION
  include)

if(ENABLE_TESTS)
enable_testing()
add_subdirectory(test)
endif()

