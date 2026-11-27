// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSPARSE_BSPGEMM_NUMERIC_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_BSPGEMM_NUMERIC_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include "KokkosSparse_Utils_mkl.hpp"
#include "KokkosSparse_SortCrs.hpp"
#include "mkl_spblas.h"
#endif

namespace KokkosSparse {
namespace Impl {

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
// Block SpGEMM numeric phase with MKL. The block sparsity pattern of C was
// already computed by spgemm_symbolic (on the block graphs, in CSR format).
// The first call creates a BSR C in the MKL handle with mkl_sparse_sp2m's
// SPARSE_STAGE_NNZ_COUNT stage, and subsequent calls reuse it, only running
// SPARSE_STAGE_FINALIZE_MULT to compute the entries and values.
template <typename KernelHandle, typename ain_row_index_view_type, typename ain_nonzero_index_view_type,
          typename ain_nonzero_value_view_type, typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
          typename bin_nonzero_value_view_type, typename cin_row_index_view_type, typename cin_nonzero_index_view_type,
          typename cin_nonzero_value_view_type>
void bspgemm_numeric_mkl(KernelHandle *handle, typename KernelHandle::nnz_lno_t m, typename KernelHandle::nnz_lno_t n,
                         typename KernelHandle::nnz_lno_t k, typename KernelHandle::nnz_lno_t blockDim,
                         ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
                         ain_nonzero_value_view_type valuesA, bin_row_index_view_type rowptrB,
                         bin_nonzero_index_view_type colidxB, bin_nonzero_value_view_type valuesB,
                         cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC,
                         cin_nonzero_value_view_type valuesC) {
  using ExecSpace   = typename KernelHandle::HandleExecSpace;
  using index_type  = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  using MKLMatrix   = MKLSparseMatrix<scalar_type>;
  if (!handle->is_symbolic_called() || !handle->are_rowptrs_computed()) {
    throw std::runtime_error(
        "KokkosSparse::bspgemm_numeric[TPL_MKL]: must first call spgemm_symbolic with the same handle.");
  }
  const size_type c_nnz = handle->get_c_nnz();
  if (c_nnz == size_type(0)) {
    handle->set_computed_entries();
    handle->set_call_numeric();
    return;
  }
  ExecSpace exec;
  MKLMatrix A = MKLMatrix::create_bsr(blockDim, m, n, const_cast<size_type *>(rowptrA.data()),
                                      const_cast<index_type *>(colidxA.data()),
                                      const_cast<scalar_type *>(valuesA.data()));
  MKLMatrix B = MKLMatrix::create_bsr(blockDim, n, k, const_cast<size_type *>(rowptrB.data()),
                                      const_cast<index_type *>(colidxB.data()),
                                      const_cast<scalar_type *>(valuesB.data()));
  matrix_descr generalDescr;
  generalDescr.type = SPARSE_MATRIX_TYPE_GENERAL;
  generalDescr.mode = SPARSE_FILL_MODE_FULL;
  generalDescr.diag = SPARSE_DIAG_NON_UNIT;
  auto mklSpgemmHandle = handle->get_mkl_spgemm_handle();
  if (!mklSpgemmHandle || mklSpgemmHandle->blockDim != blockDim) {
    // C in the handle is either missing or the CSR block graph from symbolic:
    // replace it with a BSR matrix
    sparse_matrix_t rawC;
    KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A,
                                                SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B,
                                                SPARSE_STAGE_NNZ_COUNT, &rawC));
    handle->create_mkl_spgemm_handle(rawC, blockDim);
    mklSpgemmHandle = handle->get_mkl_spgemm_handle();
  }
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A,
                                              SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B,
                                              SPARSE_STAGE_FINALIZE_MULT, &mklSpgemmHandle->C));
  A.destroy();
  B.destroy();
  MKLMatrix wrappedC(mklSpgemmHandle->C);
  MKL_INT blockDimC = 0, nrows = 0, ncols = 0;
  MKL_INT *rowptrRaw     = nullptr;
  MKL_INT *colidxRaw     = nullptr;
  scalar_type *valuesRaw = nullptr;
  wrappedC.export_bsr_data(blockDimC, nrows, ncols, rowptrRaw, colidxRaw, valuesRaw);
  // MKL's C must have exactly the structure computed by symbolic (up to the
  // order of entries within each row). Always check the sizes, since copying
  // c_nnz entries out of MKL's arrays depends on them. Only check every rowptr
  // in debug mode.
  bool sameStructure = blockDimC == blockDim && nrows == m && ncols == k && size_type(rowptrRaw[m]) == c_nnz;
#ifndef NDEBUG
  for (index_type i = 0; sameStructure && i < m; i++) {
    sameStructure = size_type(rowptrRaw[i]) == rowptrC(i);
  }
#endif
  if (!sameStructure) {
    throw std::runtime_error(
        "KokkosSparse::bspgemm_numeric[TPL_MKL]: structure of C computed by MKL does not match the symbolic phase");
  }
  Kokkos::View<index_type *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> colidxRawView(colidxRaw,
                                                                                                       c_nnz);
  Kokkos::View<scalar_type *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> valuesRawView(
      valuesRaw, c_nnz * blockDim * blockDim);
  Kokkos::deep_copy(exec, colidxC, colidxRawView);
  Kokkos::deep_copy(exec, valuesC, valuesRawView);
  // mkl_sparse_sp2m does not produce sorted output, so sort C now if the user
  // requires sorted output.
  if (handle->get_result_sorted()) {
    KokkosSparse::sort_bsr_matrix(exec, blockDim, rowptrC, colidxC, valuesC);
  }
  handle->set_call_numeric();
  handle->set_computed_entries();
}

#define BSPGEMM_NUMERIC_DECL_MKL(SCALAR, EXEC)                                                                         \
  template <bool ETI_SPEC_AVAIL>                                                                                       \
  struct BSPGEMM_NUMERIC<                                                                                              \
      KokkosKernels::Experimental::KokkosKernelsHandle<const MKL_INT, const MKL_INT, const SCALAR, EXEC,               \
                                                       Kokkos::HostSpace, Kokkos::HostSpace>,                          \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,                  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      Kokkos::View<SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                                                           \
      true, ETI_SPEC_AVAIL> {                                                                                          \
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<const MKL_INT, const MKL_INT, const SCALAR,  \
                                                                          EXEC, Kokkos::HostSpace, Kokkos::HostSpace>; \
    using c_int_view_t =                                                                                               \
        Kokkos::View<const MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,          \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                                         \
    using int_view_t = Kokkos::View<MKL_INT *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                          \
    using c_scalar_view_t =                                                                                            \
        Kokkos::View<const SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,           \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                                         \
    using scalar_view_t =                                                                                              \
        Kokkos::View<SCALAR *, KokkosKernels::default_layout, Kokkos::Device<EXEC, Kokkos::HostSpace>,                 \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                                                         \
    static void bspgemm_numeric(KernelHandle *handle, typename KernelHandle::const_nnz_lno_t m,                        \
                                typename KernelHandle::const_nnz_lno_t n, typename KernelHandle::const_nnz_lno_t k,    \
                                typename KernelHandle::const_nnz_lno_t blockDim, c_int_view_t row_mapA,                \
                                c_int_view_t entriesA, c_scalar_view_t valuesA, bool, c_int_view_t row_mapB,           \
                                c_int_view_t entriesB, c_scalar_view_t valuesB, bool, c_int_view_t row_mapC,           \
                                int_view_t &entriesC, scalar_view_t &valuesC) {                                        \
      std::string label = "KokkosSparse::bspgemm_numeric[TPL_MKL," + KokkosKernels::ArithTraits<SCALAR>::name() + "]"; \
      Kokkos::Profiling::pushRegion(label);                                                                            \
      bspgemm_numeric_mkl(handle->get_spgemm_handle(), m, n, k, blockDim, row_mapA, entriesA, valuesA, row_mapB,       \
                          entriesB, valuesB, row_mapC, entriesC, valuesC);                                             \
      Kokkos::Profiling::popRegion();                                                                                  \
    }                                                                                                                  \
  };

#define BSPGEMM_NUMERIC_DECL_MKL_E(EXEC)                 \
  BSPGEMM_NUMERIC_DECL_MKL(float, EXEC)                  \
  BSPGEMM_NUMERIC_DECL_MKL(double, EXEC)                 \
  BSPGEMM_NUMERIC_DECL_MKL(Kokkos::complex<float>, EXEC) \
  BSPGEMM_NUMERIC_DECL_MKL(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
BSPGEMM_NUMERIC_DECL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
BSPGEMM_NUMERIC_DECL_MKL_E(Kokkos::OpenMP)
#endif
#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

}  // namespace Impl
}  // namespace KokkosSparse

#endif
