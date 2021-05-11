//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_3MM kernel reference implementation:
///
/// E := A*B 
/// F := C*D 
/// G := E*F 
///
/// for (Index_type i = 0; i < NI; i++) {
///   for (Index_type j = 0; j < NJ; j++) {
///     E[i][j] = 0.0;
///     for (Index_type k = 0; k < NK; ++k) {
///       E[i][j] += A[i][k] * B[k][j];
///     }
///   }
/// } 
/// for (Index_type j = 0; j < NJ; j++) {
///   for (Index_type l = 0; l < NL; l++) {
///	F[j][l] = 0.0;
///	for (Index_type m = 0; m < NM; ++m) {
///	  F[j][l] += C[j][m] * D[m][l];
///     }
///   }
/// }
/// for (Index_type i = 0; i < NI; i++) {
///   for (Index_type l = 0; l < NL; l++) {
///     G[i][l] = 0.0;
///     for (Index_type j = 0; j < NJ; ++j) {
///	  G[i][l] += E[i][j] * F[j][l];
///     }
///   }
/// }
///

#ifndef RAJAPerf_POLYBENCH_3MM_HPP
#define RAJAPerf_POLYBENCH_3MM_HPP

#define POLYBENCH_3MM_DATA_SETUP \
  Real_ptr A = m_A; \
  Real_ptr B = m_B; \
  Real_ptr C = m_C; \
  Real_ptr D = m_D; \
  Real_ptr E = m_E; \
  Real_ptr F = m_F; \
  Real_ptr G = m_G; \
\
  const Index_type ni = m_ni; \
  const Index_type nj = m_nj; \
  const Index_type nk = m_nk; \
  const Index_type nl = m_nl; \
  const Index_type nm = m_nm;


#define POLYBENCH_3MM_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_3MM_BODY2 \
  dot += A[k + i*nk] * B[j + k*nj];

#define POLYBENCH_3MM_BODY3 \
  E[j + i*nj] = dot;

#define POLYBENCH_3MM_BODY4 \
  Real_type dot = 0.0;

#define POLYBENCH_3MM_BODY5 \
  dot += C[m + j*nm] * D[l + m*nl];

#define POLYBENCH_3MM_BODY6 \
  F[l + j*nl] = dot;

#define POLYBENCH_3MM_BODY7 \
  Real_type dot = 0.0;

#define POLYBENCH_3MM_BODY8 \
  dot += E[j + i*nj] * F[l + j*nl];

#define POLYBENCH_3MM_BODY9 \
  G[l + i*nl] = dot;


#define POLYBENCH_3MM_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_3MM_BODY2_RAJA \
  dot += Aview(i,k) * Bview(k,j);

#define POLYBENCH_3MM_BODY3_RAJA \
  Eview(i,j) = dot;

#define POLYBENCH_3MM_BODY4_RAJA \
  dot = 0.0;

#define POLYBENCH_3MM_BODY5_RAJA \
  dot += Cview(j,m) * Dview(m,l)

#define POLYBENCH_3MM_BODY6_RAJA \
  Fview(j,l) = dot;

#define POLYBENCH_3MM_BODY7_RAJA \
  dot = 0.0;

#define POLYBENCH_3MM_BODY8_RAJA \
  dot += Eview(i,j) * Fview(j,l);

#define POLYBENCH_3MM_BODY9_RAJA \
  Gview(i,l) = dot;


#define POLYBENCH_3MM_VIEWS_RAJA \
using VIEW_TYPE = RAJA::View<Real_type, \
                             RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Aview(A, RAJA::Layout<2>(ni, nk)); \
  VIEW_TYPE Bview(B, RAJA::Layout<2>(nk, nj)); \
  VIEW_TYPE Cview(C, RAJA::Layout<2>(nj, nm)); \
  VIEW_TYPE Dview(D, RAJA::Layout<2>(nm, nl)); \
  VIEW_TYPE Eview(E, RAJA::Layout<2>(ni, nj)); \
  VIEW_TYPE Fview(F, RAJA::Layout<2>(nj, nl)); \
  VIEW_TYPE Gview(G, RAJA::Layout<2>(ni, nl));

#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_3MM : public KernelBase
{
public:

  POLYBENCH_3MM(const RunParams& params);

  ~POLYBENCH_3MM();


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
  Index_type m_nm;
  Index_type m_run_reps;
  Real_ptr m_A;
  Real_ptr m_B;
  Real_ptr m_C;
  Real_ptr m_D; 
  Real_ptr m_E;
  Real_ptr m_F;
  Real_ptr m_G;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
