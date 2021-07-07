//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_GEMVER kernel reference implementation:
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
///   }
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     x[i] = x[i] + beta * A[j][i] * y[j];
///   }
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   x[i] = x[i] + z[i];
/// }
///
/// for (Index_type i = 0; i < N; i++) {
///   for (Index_type j = 0; j < N; j++) {
///     w[i] = w[i] +  alpha * A[i][j] * x[j];
///   }
/// }
///



#ifndef RAJAPerf_POLYBENCH_GEMVER_HPP
#define RAJAPerf_POLYBENCH_GEMVER_HPP

#define POLYBENCH_GEMVER_DATA_SETUP \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
  Real_ptr A = m_A; \
  Real_ptr u1 = m_u1; \
  Real_ptr v1 = m_v1; \
  Real_ptr u2 = m_u2; \
  Real_ptr v2 = m_v2; \
  Real_ptr w = m_w; \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
\
  const Index_type n = m_n;


#define POLYBENCH_GEMVER_BODY1 \
  A[j + i*n] += u1[i] * v1[j] + u2[i] * v2[j];

#define POLYBENCH_GEMVER_BODY2 \
  Real_type dot = 0.0;

#define POLYBENCH_GEMVER_BODY3 \
  dot +=  beta * A[i + j*n] * y[j];

#define POLYBENCH_GEMVER_BODY4 \
  x[i] += dot;

#define POLYBENCH_GEMVER_BODY5 \
  x[i] += z[i];

#define POLYBENCH_GEMVER_BODY6 \
  Real_type dot = w[i];

#define POLYBENCH_GEMVER_BODY7 \
  dot +=  alpha * A[j + i*n] * x[j];

#define POLYBENCH_GEMVER_BODY8 \
  w[i] = dot;


#define POLYBENCH_GEMVER_BODY1_RAJA \
  Aview(i,j) += u1view(i) * v1view(j) + u2view(i) * v2view(j);

#define POLYBENCH_GEMVER_BODY2_RAJA \
  dot = 0.0;

#define POLYBENCH_GEMVER_BODY3_RAJA \
  dot +=  beta * Aview(j,i) * yview(j);

#define POLYBENCH_GEMVER_BODY4_RAJA \
  xview(i) += dot;

#define POLYBENCH_GEMVER_BODY5_RAJA \
  xview(i) += zview(i);

#define POLYBENCH_GEMVER_BODY6_RAJA \
  dot = w[i];

#define POLYBENCH_GEMVER_BODY7_RAJA \
  dot +=  alpha * Aview(i,j) * xview(j);

#define POLYBENCH_GEMVER_BODY8_RAJA \
  wview(i) = dot;


#define POLYBENCH_GEMVER_VIEWS_RAJA \
  using VIEW_1 = RAJA::View<Real_type, \
                            RAJA::Layout<1, Index_type, 0>>; \
\
  using VIEW_2 = RAJA::View<Real_type, \
                            RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_1 u1view(u1, RAJA::Layout<1>(n)); \
  VIEW_1 v1view(v1, RAJA::Layout<1>(n)); \
  VIEW_1 u2view(u2, RAJA::Layout<1>(n)); \
  VIEW_1 v2view(v2, RAJA::Layout<1>(n)); \
  VIEW_1 wview(w, RAJA::Layout<1>(n)); \
  VIEW_1 xview(x, RAJA::Layout<1>(n)); \
  VIEW_1 yview(y, RAJA::Layout<1>(n)); \
  VIEW_1 zview(z, RAJA::Layout<1>(n)); \
  VIEW_2 Aview(A, RAJA::Layout<2>(n, n));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_GEMVER : public KernelBase
{
public:

  POLYBENCH_GEMVER(const RunParams& params);

  ~POLYBENCH_GEMVER();


  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Index_type m_n;
  Real_type m_alpha;
  Real_type m_beta;
  Real_ptr m_A;
  Real_ptr m_u1;
  Real_ptr m_v1;
  Real_ptr m_u2;
  Real_ptr m_v2;
  Real_ptr m_w;
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
