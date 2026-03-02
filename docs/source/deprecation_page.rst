Deprecations
############

..
  (Formatting to list future deprecations)
  Deprecated in Kokkos Kernels 5.X
  ================================

Deprecated in Kokkos Kernels 5.0.2
===================================

``KOKKOSKERNELS_ENABLE_HOST_ONLY``
----------------------------------

The ``KOKKOSKERNELS_ENABLE_HOST_ONLY`` macro (defined in ``KokkosKernels_config.h`` when no device backend is enabled) is deprecated with no replacement.
It is not used in Kokkos Kernels or Trilinos and will be removed in a future release.
