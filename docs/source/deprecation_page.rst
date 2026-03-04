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

``template <typename crsMat_t, typename coor_view_type, typename perm_view_type>
  void kk_extract_diagonal_blocks_crsmatrix_with_rcb_sequential(const crsMat_t &A, coor_view_type &coors,
                                                                std::vector<crsMat_t> &DiagBlk_v, perm_view_type &perm_rcb);``
----------------------------------

The ``kk_extract_diagonal_blocks_crsmatrix_with_rcb_sequential`` function with RCB run internally (i.e., with input ``coors``) is deprecated.
It is replaced with the function (same function name) which takes the pre-run RCB partition information as inputs.
