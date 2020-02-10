KOKKOSKERNELS_FIND_IMPORTED(CHOLMOD
  INTERFACE
  HEADERS   cholmod.h cholmod_core.h
  LIBRARIES libcholmod.a libamd.a libcolamd.a libsuitesparseconfig.a
)
