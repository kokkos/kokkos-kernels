INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# At some point, we may wish to make this into real "performance
# tests," in the sense that they can pass or fail.  At that point, use
# "CATEGORIES PERFORMANCE" to mark them as such.  For now, we just
# build the executables for manual use, but don't run the tests.  They
# build correctly with or without MPI, but only run them with a single
# MPI process.

ADD_SUBDIRECTORIES(graph)

TRIBITS_ADD_EXECUTABLE(
  blas1
  SOURCES blas1.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  blas1_MV
  SOURCES blas1_MV.cpp
  COMM serial mpi
  )

# These test use yaml to archive performance results
ASSERT_DEFINED(TPL_ENABLE_yaml-cpp)
IF(TPL_ENABLE_yaml-cpp)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    performance_validate
    SOURCES performance_validate.cpp
    COMM serial mpi
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    performance_demo
    SOURCES performance_demo.cpp
    COMM serial mpi
    )

ENDIF()
