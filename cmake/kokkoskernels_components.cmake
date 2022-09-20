# Define component dependencies and enable
# them selectively based on what the user
# requests.

KOKKOSKERNELS_ADD_OPTION(
        "ENABLE_ALL_COMPONENTS"
        ON
        BOOL
        "Whether to build all the library's components. Default: ON"
)

# BATCHED only depends on COMMON which
# is always enabled so nothing more needs
# to be enabled for this component.
KOKKOSKERNELS_ADD_OPTION(
        "ENABLE_BATCHED"
        OFF
        BOOL
        "Whether to build the batched component. Default: OFF"
)

# BLAS only depends on COMMON which
# is always enabled so nothing more needs
# to be enabled for this component.
KOKKOSKERNELS_ADD_OPTION(
        "ENABLE_BLAS"
        OFF
        BOOL
        "Whether to build the blas component. Default: OFF"
)

# SPARSE depends on everything else at the moment.
KOKKOSKERNELS_ADD_OPTION(
        "ENABLE_SPARSE"
        OFF
        BOOL
        "Whether to build the sparse component. Default: OFF"
)

# GRAPH depends on everything else at the moment.
KOKKOSKERNELS_ADD_OPTION(
        "ENABLE_GRAPH"
        OFF
        BOOL
        "Whether to build the graph component. Default: OFF"
)

# The user requested individual components,
# the assumption is that a full build is not
# desired and ENABLE_ALL_COMPONENETS is turned
# off.
IF (KokkosKernels_ENABLE_BATCHED OR KokkosKernels_ENABLE_BLAS
    OR KokkosKernels_ENABLE_GRAPH OR KokkosKernels_ENABLE_SPARSE)
   SET(KokkosKernels_ENABLE_ALL_COMPONENTS OFF)
ENDIF()

# Graph depends on everything else because it depends
# on Sparse at the moment, breaking that dependency will
# allow for Graph to build pretty much as a standalone
# component.
IF (KokkosKernels_ENABLE_GRAPH)
  SET(KokkosKernels_ENABLE_BATCHED ON)
  SET(KokkosKernels_ENABLE_BLAS ON)
  SET(KokkosKernels_ENABLE_SPARSE ON)
ENDIF()

# Sparse depends on everything else so no real benefit
# here unfortunately...
IF (KokkosKernels_ENABLE_SPARSE)
  SET(KokkosKernels_ENABLE_BATCHED ON)
  SET(KokkosKernels_ENABLE_BLAS ON)
  SET(KokkosKernels_ENABLE_GRAPH ON)
ENDIF()

# At this point, if ENABLE_ALL_COMPONENTS is
# still ON we need to enable all individual
# components as they are required for this
# build.
IF (KokkosKernels_ENABLE_ALL_COMPONENTS)
  SET(KokkosKernels_ENABLE_BATCHED ON)
  SET(KokkosKernels_ENABLE_BLAS ON)
  SET(KokkosKernels_ENABLE_SPARSE ON)
  SET(KokkosKernels_ENABLE_GRAPH ON)
ENDIF()
