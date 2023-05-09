# This is largely cribbed from kokkos/cmake/kokkos_arch.cmake

IF(KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS)

  IF (Kokkos_CXX_COMPILER_ID STREQUAL XL OR Kokkos_CXX_COMPILER_ID STREQUAL XLClang)
   SET(WARNING_FLAGS
     "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
     "-Wsign-compare" "-Wtype-limits" "-Wuninitialized")
     MESSAGE(STATUS "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSEIF(Kokkos_CXX_COMPILER_ID STREQUAL GNU)
    SET(WARNING_FLAGS
      "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
      "-Wsign-compare" "-Wtype-limits" "-Wuninitialized"
      "-Wimplicit-fallthrough" "-Wignored-qualifiers"
      "-Wempty-body" "-Wclobbered")
      MESSAGE(STATUS "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSEIF(Kokkos_CXX_COMPILER_ID STREQUAL Clang OR Kokkos_CXX_COMPILER_ID STREQUAL AppleClang)
    SET(WARNING_FLAGS
      "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
      "-Wsign-compare" "-Wtype-limits" "-Wuninitialized"
      "-Wimplicit-fallthrough")
    MESSAGE(STATUS "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSEIF (Kokkos_CXX_COMPILER_ID STREQUAL Intel OR Kokkos_CXX_COMPILER_ID STREQUAL IntelLLVM)
   SET(WARNING_FLAGS
     "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
     "-Wsign-compare" "-Wtype-limits" "-Wuninitialized"
     "-diag-disable=1011" "-diag-disable=869")
     MESSAGE(INFO "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for\
 Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSEIF (Kokkos_CXX_COMPILER_ID STREQUAL NVIDIA)
    SET(WARNING_FLAGS
      "-Wall" "-Wunused-parameter" "-Wshadow" "-pedantic"
      "-Wsign-compare" "-Wtype-limits" "-Wuninitialized")
    MESSAGE(STATUS "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSEIF (Kokkos_CXX_COMPILER_ID STREQUAL ARMClang)
    SET(WARNING_FLAGS
      "-Wall" "-Wshadow" "-pedantic"
      "-Wsign-compare" "-Wtype-limits" "-Wuninitialized")
    MESSAGE(STATUS "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS: adding flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}: ${WARNINGS_FLAGS}")
  ELSE()
    MESSAGE(WARNING "KOKKOSKERNELS_ENABLE_COMPILER_WARNINGS set, but don't know how to add warning flags for Kokkos_CXX_COMPILER_ID=${Kokkos_CXX_COMPILER_ID}. If you are compiling with a supported compiler, please report this issue.")
  ENDIF()



  IF(Kokkos_ENABLE_LIBQUADMATH)
    # warning: non-standard suffix on floating constant [-Wpedantic]
    LIST(REMOVE_ITEM WARNING_FLAGS "-pedantic")
  ENDIF()

  # OpenMPTarget compilers give erroneous warnings about sign comparison in loops
  IF(KOKKOS_ENABLE_OPENMPTARGET)
    LIST(REMOVE_ITEM WARNING_FLAGS "-Wsign-compare")
  ENDIF()

  STRING(REPLACE ";" " " WARNING_FLAGS "${WARNING_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
ENDIF()

