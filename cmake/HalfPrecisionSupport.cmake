# Check whether Kokkos has half precision headers
IF(EXISTS ${Kokkos_DIR}/../../../include/Kokkos_Half.hpp)
  SET(HAVE_KOKKOS_HALFMATH TRUE)
ELSE()
  SET(HAVE_KOKKOS_HALFMATH FALSE)
ENDIF()
