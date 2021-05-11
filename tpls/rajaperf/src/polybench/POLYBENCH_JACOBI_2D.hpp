//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_JACOBI_2D kernel reference implementation:
///
/// for (t = 0; t < TSTEPS; t++)
/// {
///   for (i = 1; i < N - 1; i++) {
///     for (j = 1; j < N - 1; j++) {
///       B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]);
///     }
///   }
///   for (i = 1; i < N - 1; i++) {
///     for (j = 1; j < N - 1; j++) {
///       A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j]);
///     }
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_JACOBI_2D_HPP
#define RAJAPerf_POLYBENCH_JACOBI_2D_HPP

#define POLYBENCH_JACOBI_2D_DATA_SETUP \
  Real_ptr A = m_Ainit; \
  Real_ptr B = m_Binit; \
\
  const Index_type N = m_N; \
  const Index_type tsteps = m_tsteps;

#define POLYBENCH_JACOBI_2D_DATA_RESET \
  m_Ainit = m_A; \
  m_Binit = m_B; \
  m_A = A; \
  m_B = B;


#define POLYBENCH_JACOBI_2D_BODY1 \
  B[j + i*N] = 0.2 * (A[j + i*N] + A[j-1 + i*N] + A[j+1 + i*N] + A[j + (i+1)*N] + A[j + (i-1)*N]);

#define POLYBENCH_JACOBI_2D_BODY2 \
  A[j + i*N] = 0.2 * (B[j + i*N] + B[j-1 + i*N] + B[j+1 + i*N] + B[j + (i+1)*N] + B[j + (i-1)*N]);


#define POLYBENCH_JACOBI_2D_BODY1_RAJA \
  Bview(i,j) = 0.2 * (Aview(i,j) + Aview(i,j-1) + Aview(i,j+1) + Aview(i+1,j) + Aview(i-1,j));

#define POLYBENCH_JACOBI_2D_BODY2_RAJA \
  Aview(i,j) = 0.2 * (Bview(i,j) + Bview(i,j-1) + Bview(i,j+1) + Bview(i+1,j) + Bview(i-1,j));


#define POLYBENCH_JACOBI_2D_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(N, N)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_JACOBI_2D : public KernelBase
{
public:

  POLYBENCH_JACOBI_2D(const RunParams& params);

  ~POLYBENCH_JACOBI_2D();


  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_N;
  Index_type m_tsteps;

  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_Ainit;
  Real_ptr m_Binit;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
