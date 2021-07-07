//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_2MM kernel reference implementation:
///
/// D := alpha*A*B*C + beta*D
///
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type j = 0; j < nj; j++) {
///     tmp[i][j] = 0.0;
///     for (Index_type k = 0; k < nk; ++k) {
///       tmp[i][j] += alpha * A[i][k] * B[k][j];
///     }
///   }
/// } 
/// for (Index_type i = 0; i < ni; i++) {
///   for (Index_type l = 0; l < nl; l++) {
///     D[i][l] *= beta;  // NOTE: Changed to 'D[i][l] = beta;' 
///                       // to avoid need for memset operation
///                       // to zero out matrix.
///     for (Index_type j = 0; j < nj; ++j) {
///       D[i][l] += tmp[i][j] * C[j][l];
///     } 
///   }
/// } 
///


#ifndef RAJAPerf_POLYBENCH_2MM_HPP
#define RAJAPerf_POLYBENCH_2MM_HPP


#define POLYBENCH_2MM_DATA_SETUP \
  Real_ptr tmp = m_tmp; \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
  const Index_type nl = m_nl;


#define POLYBENCH_2MM_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_2MM_BODY2 \
  dot += alpha * A[k + i*nk] * B[j + k*nj];

#define POLYBENCH_2MM_BODY3 \
  tmp[j + i*nj] = dot;

#define POLYBENCH_2MM_BODY4 \
  Real_type dot = beta;

#define POLYBENCH_2MM_BODY5 \
  dot += tmp[j + i*nj] * C[l + j*nl];

#define POLYBENCH_2MM_BODY6 \
  D[l + i*nl] = dot;


#define POLYBENCH_2MM_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_2MM_BODY2_RAJA \
  dot += alpha * Aview(i,k) * Bview(k,j);

#define POLYBENCH_2MM_BODY3_RAJA \
  tmpview(i,j) = dot;

#define POLYBENCH_2MM_BODY4_RAJA \
  dot = beta;

#define POLYBENCH_2MM_BODY5_RAJA \
  dot += tmpview(i,j) * Cview(j, l);

#define POLYBENCH_2MM_BODY6_RAJA \
  Dview(i,l) = dot;


#define POLYBENCH_2MM_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE tmpview(tmp, RAJA::Layout<2>(ni, nj)); \
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(nj, nl)); \
  VIEW_TYPE Dview(D, RAJA::Layout<2>(ni, nl));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_2MM : public KernelBase
{
public:

  POLYBENCH_2MM(const RunParams& params);

  ~POLYBENCH_2MM();


  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nl;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_tmp;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
