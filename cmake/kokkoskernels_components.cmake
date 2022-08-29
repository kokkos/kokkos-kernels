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


# The user requested individual components,
# the assumption is that a full build is not
# desired and ENABLE_ALL_COMPONENETS is turned
# off.
IF (KokkosKernels_ENABLE_BATCHED OR KokkosKernels_ENABLE_BLAS)
   SET(KokkosKernels_ENABLE_ALL_COMPONENTS OFF)
ENDIF()



# At this point, if ENABLE_ALL_COMPONENTS is
# still ON we need to enable all individual
# components as they are required for this
# build.
IF (KokkosKernels_ENABLE_ALL_COMPONENTS)
  SET(KokkosKernels_ENABLE_BATCHED ON)
  SET(KokkosKernels_ENABLE_BLAS ON)
  SET(KokkosKernels_ENABLE_REMAINDER ON)
ENDIF()

MESSAGE("Kokkos Kernels components")
MESSAGE("   COMMON:    ON")
MESSAGE("   BATCHED:   ${KokkosKernels_ENABLE_BATCHED}")
MESSAGE("   BLAS:      ${KokkosKernels_ENABLE_BLAS}")
MESSAGE("   REMAINDER: ${KokkosKernels_ENABLE_REMAINDER}")
