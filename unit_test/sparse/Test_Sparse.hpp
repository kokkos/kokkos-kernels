#include "Test_Sparse_block_gauss_seidel.hpp"
#include "Test_Sparse_CrsMatrix.hpp"
#include "Test_Sparse_BlockCrsMatrix.hpp"
#if !defined(KOKKOSKERNELS_CUDA_SPARSE_TESTS) || (defined(KOKKOSKERNELS_CUDA_SPARSE_TESTS) && defined(KOKKOS_ENABLE_CUDA_UVM))
#include "Test_Sparse_findRelOffset.hpp"
#endif
#include "Test_Sparse_gauss_seidel.hpp"
#include "Test_Sparse_replaceSumInto.hpp"
#include "Test_Sparse_replaceSumIntoLonger.hpp"
#include "Test_Sparse_spadd.hpp"
#include "Test_Sparse_spgemm_jacobi.hpp"
#include "Test_Sparse_spgemm.hpp"
#include "Test_Sparse_spiluk.hpp"
#include "Test_Sparse_spmv.hpp"
#include "Test_Sparse_sptrsv.hpp"
#if !defined(KOKKOSKERNELS_CUDA_SPARSE_TESTS) || (defined(KOKKOSKERNELS_CUDA_SPARSE_TESTS) && defined(KOKKOS_ENABLE_CUDA_UVM))
#include "Test_Sparse_trsv.hpp"
#endif

#if defined(KOKKOSKERNELS_CUDA_SPARSE_TESTS) && defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
#include "Test_Sparse_Utils_cusparse.hpp"
#endif
