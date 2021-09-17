/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef TEST_SPARSE_SPMV_BSR_HPP
#define TEST_SPARSE_SPMV_BSR_HPP

//#include "KokkosKernels_ETIHelperMacros.h"
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include "KokkosSparse_BsrMatrix.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_Test_Structured_Matrix.hpp>

typedef Kokkos::complex<double> kokkos_complex_double;
typedef Kokkos::complex<float> kokkos_complex_float;

namespace Test_Bsr {

/// Random generator
template<typename Scalar>
inline Scalar random() {
  auto const max = static_cast<Scalar>(RAND_MAX) + static_cast<Scalar>(1);
  return static_cast<Scalar>(std::rand()) / max;
}

template<typename Scalar>
inline void set_random_value(Scalar &v) {
  v = random<Scalar>();
}

template <typename Scalar>
inline void set_random_value(Kokkos::complex<Scalar> &v) {
  Scalar vre = random<Scalar>();
  Scalar vim = random<Scalar>();
  v = Kokkos::complex<Scalar>(vre, vim);
}

template<typename Scalar>
inline void set_random_value(std::complex<Scalar> &v) {
  Scalar vre = random<Scalar>();
  Scalar vim = random<Scalar>();
  v = std::complex<Scalar>(vre, vim);
}

/// \brief Driver routine for checking BsrMatrix times vector
template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void check_bsrm_times_v(const char fOp[], scalar_t alpha, scalar_t beta,
                        const lno_t bMax, int &num_errors
) 
{

  // The mat_structure view is used to generate a matrix using
  // finite difference (FD) or finite element (FE) discretization
  // on a cartesian grid.
  Kokkos::View<lno_t *[3], Kokkos::HostSpace> mat_structure(
      "Matrix Structure", 3);
  mat_structure(0, 0) = 8;   // Request 8 grid point in 'x' direction
  mat_structure(0, 1) = 0;   // Add BC to the left
  mat_structure(0, 2) = 0;   // Add BC to the right
  mat_structure(1, 0) = 7;   // Request 7 grid point in 'y' direction
  mat_structure(1, 1) = 0;   // Add BC to the bottom
  mat_structure(1, 2) = 0;   // Add BC to the top
  mat_structure(2, 0) = 9;   // Request 9 grid point in 'z' direction
  mat_structure(2, 1) = 0;   // Add BC to the bottom
  mat_structure(2, 2) = 0;   // Add BC to the top

  typedef
      typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>
          crsMat_t;
  typedef typename crsMat_t::values_type::non_const_type scalar_view_t;
  typedef scalar_view_t x_vector_type;
  typedef scalar_view_t y_vector_type;

  crsMat_t mat_b1 =
      Test::generate_structured_matrix3D<crsMat_t>("FD", mat_structure);

for (int ir = 0; ir < mat_b1.numRows(); ++ir) {
  const auto jbeg = mat_b1.graph.row_map(ir);
  const auto jend = mat_b1.graph.row_map(ir + 1);
  std::set< lno_t > list;
  for (auto jj = jbeg; jj < jend; ++jj) {
    list.insert( mat_b1.graph.entries(jj) );
  }
  if (list.size() < jend - jbeg) {
    std::cout << " Duplicate Entry on row " << ir << " !!! \n";
    exit(-123);
  }
}

  num_errors = 0;
  for (lno_t blockSize = 1; blockSize <= bMax; ++blockSize) {

    //
    // Fill blocks with random values
    //

    lno_t nRow   = blockSize * mat_b1.numRows();
    lno_t nCol   = blockSize * mat_b1.numCols();
    size_type nnz = static_cast<size_type>(blockSize)
                    * static_cast<size_type>(blockSize) * mat_b1.nnz();

    std::vector<scalar_t> mat_val(nnz);
    for (size_type ii = 0; ii < nnz; ++ii)
      set_random_value(mat_val[ii]);

    //
    // Create graph for CrsMatrix
    //

    std::vector<lno_t> mat_rowmap(nRow + 1, 0);
    std::vector<lno_t> mat_colidx(nnz, 0);

    auto *rowmap = &mat_rowmap[0];
    auto *cols = &mat_colidx[0];

    for (lno_t ir = 0; ir < mat_b1.numRows(); ++ir) {
      const auto jbeg = mat_b1.graph.row_map(ir);
      const auto jend = mat_b1.graph.row_map(ir + 1);
      for (lno_t ib = 0; ib < blockSize; ++ib) {
        const lno_t my_row = ir * blockSize + ib;
        rowmap[my_row + 1] = rowmap[my_row] + (jend - jbeg) * blockSize;
        for (lno_t ijk = jbeg; ijk < jend; ++ijk) {
          const auto col0 = mat_b1.graph.entries(ijk);
          for (lno_t jb = 0; jb < blockSize; ++jb) {
            cols[rowmap[my_row] + (ijk - jbeg) * blockSize + jb] = col0 * blockSize + jb;
          }
        }
      }
    }  // for (lno_t ir = 0; ir < mat_b1.numRows(); ++ir)

    // Create the CrsMatrix for the reference computation
    crsMat_t Acrs("new_crs_matr", nRow, nCol, nnz, &mat_val[0], &mat_rowmap[0], &mat_colidx[0]);

    x_vector_type xref("new_right_hand_side", nRow);
    for (lno_t ir = 0; ir < nRow; ++ir)
      set_random_value(xref(ir));

    y_vector_type y0("y_init", nRow);
    for (lno_t ir = 0; ir < nRow; ++ir)
      set_random_value(y0(ir));

    y_vector_type ycrs("crs_product_result", nRow);
    for (lno_t ir = 0; ir < nRow; ++ir)
      ycrs(ir) = y0(ir);

    KokkosSparse::spmv(fOp, alpha, Acrs, xref, beta, ycrs);

    y_vector_type ybsr("bsr_product_result", nRow);
    for (lno_t ir = 0; ir < nRow; ++ir)
      ybsr(ir) = y0(ir);

    // Create the BsrMatrix for the check test
    KokkosSparse::Experimental::BsrMatrix<scalar_t, lno_t, device, void, size_type> Absr(
        Acrs, blockSize);
    KokkosSparse::spmv(fOp, alpha, Absr, xref, beta, ybsr);

    // Compare the norms between the two products
    double error = 0.0, maxNorm = 0.0;
    for (lno_t ir = 0; ir < nRow; ++ir) {
      error = std::max<double>(
          error, Kokkos::ArithTraits<scalar_t>::abs(ycrs(ir) - ybsr(ir)));
      maxNorm = std::max<double>(maxNorm,
                                 Kokkos::ArithTraits<scalar_t>::abs(ycrs(ir)));
    }
    //
    // --- Factor ((nnz / nRow) + 1) = Average number of non-zeros per row
    //
    const auto tol = ((nnz / nRow) + 1) *
        static_cast<double>(Kokkos::ArithTraits<scalar_t>::abs(
                    Kokkos::ArithTraits<scalar_t>::epsilon() ));

    if (error > tol * maxNorm) {
      std::cout << " BSR - SpMV times V >> blockSize " << blockSize << " ratio " << error / maxNorm 
                << " tol " << tol << " maxNorm " << maxNorm
                << " alpha " << alpha << " beta " << beta
                << "\n";
      num_errors += 1;
    }

  } // for (int blockSize = 1; blockSize <= bMax; ++blockSize)

}

/// \brief Driver routine for checking BsrMatrix times multiple vector
template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void check_bsrm_times_mv(const char fOp[], scalar_t alpha, scalar_t beta,
                         const lno_t bMax, int &num_errors
) {

  // The mat_structure view is used to generate a matrix using
  // finite difference (FD) or finite element (FE) discretization
  // on a cartesian grid.
  Kokkos::View<lno_t *[3], Kokkos::HostSpace> mat_structure(
      "Matrix Structure", 3);
  mat_structure(0, 0) = 7;  // Request 7 grid point in 'x' direction
  mat_structure(0, 1) = 0;   // Add BC to the left
  mat_structure(0, 2) = 0;   // Add BC to the right
  mat_structure(1, 0) = 5;  // Request 11 grid point in 'y' direction
  mat_structure(1, 1) = 0;   // Add BC to the bottom
  mat_structure(1, 2) = 0;   // Add BC to the top
  mat_structure(2, 0) = 9;  // Request 13 grid point in 'y' direction
  mat_structure(2, 1) = 0;   // Add BC to the bottom
  mat_structure(2, 2) = 0;   // Add BC to the top

  typedef
      typename KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>
          crsMat_t;
  typedef Kokkos::View<scalar_t**, Kokkos::LayoutLeft, device> block_vector_t;

  crsMat_t mat_b1 =
      Test::generate_structured_matrix3D<crsMat_t>("FD", mat_structure);

  num_errors = 0;
  const int nrhs = 5;

  for (lno_t blockSize = 1; blockSize <= bMax; ++blockSize) {

    //
    // Fill blocks with random values
    //

    lno_t nRow   = blockSize * mat_b1.numRows();
    lno_t nCol   = blockSize * mat_b1.numCols();
    size_type nnz = static_cast<size_type>(blockSize)
                    * static_cast<size_type>(blockSize) * mat_b1.nnz();

    std::vector<scalar_t> mat_val(nnz);
    for (size_type ii = 0; ii < nnz; ++ii)
      set_random_value(mat_val[ii]);

    //
    // Create graph for CrsMatrix
    //

    std::vector<lno_t> mat_rowmap(nRow + 1);
    std::vector<lno_t> mat_colidx(nnz);

    mat_rowmap.resize(nRow + 1);
    auto *rowmap = &mat_rowmap[0];
    rowmap[0]   = 0;

    mat_colidx.resize(nnz);
    auto *cols = &mat_colidx[0];

    for (lno_t ir = 0; ir < mat_b1.numRows(); ++ir) {
      auto mat_b1_row = mat_b1.rowConst(ir);
      for (lno_t ib = 0; ib < blockSize; ++ib) {
        lno_t my_row         = ir * blockSize + ib;
        rowmap[my_row + 1] = rowmap[my_row] + mat_b1_row.length * blockSize;
        for (lno_t ijk = 0; ijk < mat_b1_row.length; ++ijk) {
          auto col0 = mat_b1_row.colidx(ijk);
          for (lno_t jb = 0; jb < blockSize; ++jb) {
            cols[rowmap[my_row] + ijk * blockSize + jb] = col0 * blockSize + jb;
          }
        }
      }
    }  // for (lno_t ir = 0; ir < mat_b1.numRows(); ++ir)

    // Create the CrsMatrix for the reference computation
    crsMat_t Acrs("new_crs_matr", nRow, nCol, nnz, &mat_val[0], rowmap, cols);

    block_vector_t xref("new_right_hand_side", nRow, nrhs);
    for (int jc = 0; jc < nrhs; ++jc)
      for (lno_t ir = 0; ir < nRow; ++ir)
        set_random_value(xref(ir, jc));

    block_vector_t y0("y_init", nRow, nrhs);
    for (int jc = 0; jc < nrhs; ++jc)
      for (lno_t ir = 0; ir < nRow; ++ir)
        set_random_value(y0(ir, jc));

    block_vector_t ycrs("crs_product_result", nRow, nrhs);
    for (int jc = 0; jc < nrhs; ++jc)
      for (int ir = 0; ir < nRow; ++ir)
        ycrs(ir, jc) = y0(ir, jc);

    KokkosSparse::spmv(fOp, alpha, Acrs, xref, beta, ycrs);

    block_vector_t ybsr("bsr_product_result", nRow, nrhs);
    for (int jc = 0; jc < nrhs; ++jc)
      for (int ir = 0; ir < nRow; ++ir)
        ybsr(ir, jc) = y0(ir, jc);

    // Create the BsrMatrix for the check test
    KokkosSparse::Experimental::BsrMatrix<scalar_t, lno_t, device, void, size_type> Absr(Acrs, blockSize);
    KokkosSparse::spmv(fOp, alpha, Absr, xref, beta, ybsr);

    // Compare the norms between the two products
    double error = 0.0, maxNorm = 0.0;
    for (int jc = 0; jc < nrhs; ++jc) {
      for (int ir = 0; ir < nRow; ++ir) {
        error = std::max<double>(
            error, Kokkos::ArithTraits<scalar_t>::abs(ycrs(ir, jc) - ybsr(ir, jc)));
        maxNorm = std::max<double>(maxNorm,
                                   Kokkos::ArithTraits<scalar_t>::abs(ycrs(ir, jc)));
      }
    }
    auto tol = ((nnz / nRow) + 1) * static_cast<double>(Kokkos::ArithTraits<scalar_t>::abs(
        Kokkos::ArithTraits<scalar_t>::epsilon() ));
    if (error > tol * maxNorm) {
      std::cout << " BSR - SpMV times MV >> blockSize " << blockSize << " ratio " << error / maxNorm 
                << " tol " << tol << " maxNorm " << maxNorm
                << " alpha " << alpha << " beta " << beta
                << "\n";
      num_errors += 1;
    }

  } // for (int blockSize = 1; blockSize <= bMax; ++blockSize)

}

} // namespace Test_Bsr


template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void testSpMVBsrMatrix() {

  std::vector<char> modes = {'N', 'C', 'T', 'H'};
  std::vector<double> testAlphaBeta = {0.0, 0.0, 
                                       -1.0, 0.0,
                                       0.0, 1.0,
                                       3.1, -2.5};

  const lno_t bMax = 13;

  //--- Test single vector case
  for(const auto mode : modes)
  {
    int num_errors = 0;
    for(size_t ii = 0; ii < testAlphaBeta.size(); ii += 2)
    {
      auto alpha_s = static_cast<scalar_t>(testAlphaBeta[ii]);
      auto beta_s = static_cast<scalar_t>(testAlphaBeta[ii+1]);
      num_errors = 0;
      Test_Bsr::check_bsrm_times_v< scalar_t, lno_t, size_type, device>(&mode, alpha_s, 
                           beta_s, bMax, num_errors);
      if (num_errors>0) {
        printf("KokkosSparse::Test::spmv_bsr: %i errors of %i with params: "
               "%c %lf %lf\n", num_errors, bMax,
               mode, Kokkos::ArithTraits<scalar_t>::abs(alpha_s), Kokkos::ArithTraits<scalar_t>::abs(beta_s));
      }
      EXPECT_TRUE(num_errors==0);
    }
  }

}

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void testBsrMatrix_SpM_MV() {

  std::vector<char> modes = {'N', 'C', 'T', 'H'};
  std::vector<double> testAlphaBeta = {0.0, 0.0, 
                                       -1.0, 0.0,
                                       0.0, 1.0,
                                       3.1, -2.5};

  const lno_t bMax = 13;

  //--- Test multiple vector case
  for(auto mode : modes)
  {
    int num_errors = 0;
    for(size_t ii = 0; ii < testAlphaBeta.size(); ii += 2)
    {
      auto alpha_s = static_cast<scalar_t>(testAlphaBeta[ii]);
      auto beta_s = static_cast<scalar_t>(testAlphaBeta[ii+1]);
      num_errors = 0;
      Test_Bsr::check_bsrm_times_mv< scalar_t, lno_t, size_type, device>(&mode, alpha_s,
                           beta_s, bMax, num_errors);
      if (num_errors>0) {
        printf("KokkosSparse::Test::spm_mv_bsr: %i errors of %i with params: "
               "%c %lf %lf\n", num_errors, bMax,
               mode, Kokkos::ArithTraits<scalar_t>::abs(alpha_s), Kokkos::ArithTraits<scalar_t>::abs(beta_s));
      }
      EXPECT_TRUE(num_errors==0);
    }
  }

}

//////////////////////////

#define EXECUTE_BSR_TIMES_VEC_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)                     \
  TEST_F(TestCategory,                                                        \
         sparse##_##bsrmat_times_vec##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    testSpMVBsrMatrix<SCALAR, ORDINAL, OFFSET, DEVICE>();                     \
  }

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&      \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(double, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||     \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&         \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&    \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&           \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||  \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&       \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(float, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&           \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||     \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&    \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&           \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&           \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||  \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(float, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&            \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||            \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&        \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||            \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&            \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||         \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&        \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||         \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&           \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||           \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&       \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||           \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&           \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||        \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&       \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||        \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_VEC_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif

#undef EXECUTE_BSR_TIMES_VEC_TEST

//////////////////////////

#define EXECUTE_BSR_TIMES_MVEC_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)                     \
  TEST_F(TestCategory,                                                        \
         sparse##_##bsrmat_times_multivec##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    testBsrMatrix_SpM_MV<SCALAR, ORDINAL, OFFSET, DEVICE>();                     \
  }

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&      \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(double, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||     \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&         \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&    \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&           \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_DOUBLE) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||  \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(double, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&       \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&        \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(float, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&           \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||     \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(float, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&          \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&    \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) || \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&           \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(float, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_FLOAT) &&           \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) && \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||  \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&            \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(float, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&            \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||            \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_double, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&        \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||            \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_double, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&            \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||         \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_double, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&        \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||         \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                   \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_double, int64_t, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&           \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||           \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_float, int, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&       \
     defined(KOKKOSKERNELS_INST_OFFSET_INT)) ||           \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_float, int64_t, int, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT) &&           \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||        \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_float, int, size_t, TestExecSpace)
#endif

#if (defined(KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_) && \
     defined(KOKKOSKERNELS_INST_ORDINAL_INT64_T) &&       \
     defined(KOKKOSKERNELS_INST_OFFSET_SIZE_T)) ||        \
    (!defined(KOKKOSKERNELS_ETI_ONLY) &&                  \
     !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
EXECUTE_BSR_TIMES_MVEC_TEST(kokkos_complex_float, int64_t, size_t, TestExecSpace)
#endif

#undef EXECUTE_BSR_TIMES_MVEC_TEST

#endif


