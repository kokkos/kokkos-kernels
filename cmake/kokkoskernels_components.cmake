# Define component dependencies and enable
# them selectively based on what the user
# requests.
#
# Each KokkosKernels_ENABLE_COMPONENT_<NAME> option is tri-state:
#   ON    the user explicitly requested the component
#   OFF   the user explicitly disabled the component
#   ""    (default) follow KokkosKernels_ENABLE_ALL_COMPONENTS
#
# The cache entries only hold what the user asked for and are never
# overwritten here. The resolved values are stored in normal variables of
# the same name, which shadow the cache entries for the rest of the
# configure. This keeps reconfiguring an existing build directory
# predictable: no value forced by a previous run can stick around.

set(KOKKOSKERNELS_COMPONENTS BATCHED BLAS LAPACK SPARSE GRAPH ODE)

# Components that need other components to build. The lists are
# transitively closed so a single pass over them is enough.
# BATCHED, BLAS and LAPACK only depend on COMMON which is always enabled.
# ODE pulls the few BATCHED headers it needs by itself.
# SPARSE depends on everything else at the moment, and so does GRAPH
# because it depends on SPARSE. Breaking that dependency will allow
# GRAPH to build pretty much as a standalone component.
set(KOKKOSKERNELS_COMPONENT_SPARSE_DEPENDS BATCHED BLAS LAPACK GRAPH)
set(KOKKOSKERNELS_COMPONENT_GRAPH_DEPENDS BATCHED BLAS LAPACK SPARSE)

kokkoskernels_add_option("ENABLE_ALL_COMPONENTS" ON BOOL
  "Whether to build the components whose KokkosKernels_ENABLE_COMPONENT_* option is left empty. Default: ON")

foreach(COMPONENT ${KOKKOSKERNELS_COMPONENTS})
  string(TOLOWER ${COMPONENT} COMPONENT_LC)
  kokkoskernels_add_option("ENABLE_COMPONENT_${COMPONENT}" "" STRING
    "Whether to build the ${COMPONENT_LC} component: ON, OFF, or empty to follow KokkosKernels_ENABLE_ALL_COMPONENTS. Default: empty")
  set_property(CACHE KokkosKernels_ENABLE_COMPONENT_${COMPONENT} PROPERTY STRINGS ";ON;OFF")

  # Normalize what the user requested to ON, OFF or "" (not set)
  set(REQUESTED "${KokkosKernels_ENABLE_COMPONENT_${COMPONENT}}")
  string(TOUPPER "${REQUESTED}" REQUESTED_UC)
  if(REQUESTED STREQUAL "")
    set(KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED "")
  elseif(REQUESTED_UC MATCHES "^(ON|TRUE|YES|Y|1)$")
    set(KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED ON)
  elseif(REQUESTED_UC MATCHES "^(OFF|FALSE|NO|N|0)$")
    set(KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED OFF)
  else()
    message(FATAL_ERROR
      "Invalid value '${REQUESTED}' for KokkosKernels_ENABLE_COMPONENT_${COMPONENT}. "
      "Valid values are ON, OFF, or empty to follow KokkosKernels_ENABLE_ALL_COMPONENTS.")
  endif()

  if(KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED STREQUAL "")
    if(KokkosKernels_ENABLE_ALL_COMPONENTS)
      set(KokkosKernels_ENABLE_COMPONENT_${COMPONENT} ON)
    else()
      set(KokkosKernels_ENABLE_COMPONENT_${COMPONENT} OFF)
    endif()
  else()
    set(KokkosKernels_ENABLE_COMPONENT_${COMPONENT} ${KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED})
  endif()
  # Extra information printed next to the component in the configure summary
  set(KOKKOSKERNELS_COMPONENT_${COMPONENT}_NOTE "")
endforeach()

# An explicitly disabled component turns off the components that need it,
# unless they were explicitly requested, in which case the request cannot
# be satisfied and we stop here rather than ignore one of the two options.
foreach(COMPONENT ${KOKKOSKERNELS_COMPONENTS})
  foreach(DEPENDENCY ${KOKKOSKERNELS_COMPONENT_${COMPONENT}_DEPENDS})
    if(KOKKOSKERNELS_COMPONENT_${DEPENDENCY}_REQUESTED STREQUAL "OFF")
      if(KOKKOSKERNELS_COMPONENT_${COMPONENT}_REQUESTED STREQUAL "ON")
        list(JOIN KOKKOSKERNELS_COMPONENT_${COMPONENT}_DEPENDS ", " DEPENDENCIES)
        message(FATAL_ERROR
          "KokkosKernels_ENABLE_COMPONENT_${COMPONENT}=ON requires ${DEPENDENCY}, "
          "but KokkosKernels_ENABLE_COMPONENT_${DEPENDENCY}=OFF. "
          "${COMPONENT} depends on: ${DEPENDENCIES}.")
      endif()
      if(KokkosKernels_ENABLE_COMPONENT_${COMPONENT})
        message(STATUS "Disabling component ${COMPONENT} because it requires ${DEPENDENCY}, which is disabled")
        set(KokkosKernels_ENABLE_COMPONENT_${COMPONENT} OFF)
        set(KOKKOSKERNELS_COMPONENT_${COMPONENT}_NOTE "(requires ${DEPENDENCY})")
      endif()
    endif()
  endforeach()
endforeach()

# Enable the dependencies of the remaining components. None of them can be
# explicitly disabled at this point.
foreach(COMPONENT ${KOKKOSKERNELS_COMPONENTS})
  if(KokkosKernels_ENABLE_COMPONENT_${COMPONENT})
    foreach(DEPENDENCY ${KOKKOSKERNELS_COMPONENT_${COMPONENT}_DEPENDS})
      if(NOT KokkosKernels_ENABLE_COMPONENT_${DEPENDENCY})
        message(STATUS "Enabling component ${DEPENDENCY} because ${COMPONENT} requires it")
        set(KokkosKernels_ENABLE_COMPONENT_${DEPENDENCY} ON)
        set(KOKKOSKERNELS_COMPONENT_${DEPENDENCY}_NOTE "(required by ${COMPONENT})")
      endif()
    endforeach()
  endif()
endforeach()

# KOKKOSKERNELS_ALL_COMPONENTS_ENABLED says whether all components are on,
# regardless of how this came to be
# this is in the cache so we can use it as a global variable,
# but marking it as advanced should hide it from GUIs
set(ALL_COMPONENTS_ENABLED ON)
foreach(COMPONENT ${KOKKOSKERNELS_COMPONENTS})
  if(NOT KokkosKernels_ENABLE_COMPONENT_${COMPONENT})
    set(ALL_COMPONENTS_ENABLED OFF)
  endif()
endforeach()
set(KOKKOSKERNELS_ALL_COMPONENTS_ENABLED ${ALL_COMPONENTS_ENABLED} CACHE BOOL "" FORCE)
mark_as_advanced(FORCE KOKKOSKERNELS_ALL_COMPONENTS_ENABLED)
# Now that component enables are finalized, also set upper-case
# versions of component enables for the config.h

set(KOKKOSKERNELS_ENABLE_COMPONENT_BATCHED ${KokkosKernels_ENABLE_COMPONENT_BATCHED})
set(KOKKOSKERNELS_ENABLE_COMPONENT_BLAS    ${KokkosKernels_ENABLE_COMPONENT_BLAS})
set(KOKKOSKERNELS_ENABLE_COMPONENT_LAPACK  ${KokkosKernels_ENABLE_COMPONENT_LAPACK})
set(KOKKOSKERNELS_ENABLE_COMPONENT_GRAPH   ${KokkosKernels_ENABLE_COMPONENT_GRAPH})
set(KOKKOSKERNELS_ENABLE_COMPONENT_SPARSE  ${KokkosKernels_ENABLE_COMPONENT_SPARSE})
set(KOKKOSKERNELS_ENABLE_COMPONENT_ODE     ${KokkosKernels_ENABLE_COMPONENT_ODE})
