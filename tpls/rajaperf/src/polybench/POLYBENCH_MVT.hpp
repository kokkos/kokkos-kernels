//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_MVT kernel reference implementation:
///
/// for (int i = 0; i < N; i++) {
///   for (int j = 0; j < N; j++) {
///     x1[i] += A[i][j] * y1[j];
///   }
/// }
/// for (int i = 0; i < N; i++) {
///   for (int j = 0; j < N; j++) {
///     x2[i] += A[j][i] * y2[i];
///   }
/// }


#ifndef RAJAPerf_POLYBENCH_MVT_HPP
#define RAJAPerf_POLYBENCH_MVT_HPP

#define POLYBENCH_MVT_DATA_SETUP \
  Real_ptr x1 = m_x1; \
  Real_ptr x2 = m_x2; \
  Real_ptr y1 = m_y1; \
  Real_ptr y2 = m_y2; \
  Real_ptr A = m_A; \
  const Index_type N = m_N;


#define POLYBENCH_MVT_BODY1 \
  Real_type dot = 0.0;

#define POLYBENCH_MVT_BODY2 \
  dot += A[j + i*N] * y1[j];

#define POLYBENCH_MVT_BODY3 \
  x1[i] += dot;

#define POLYBENCH_MVT_BODY4 \
  Real_type dot = 0.0;

#define POLYBENCH_MVT_BODY5 \
  dot += A[i + j*N] * y2[i];

#define POLYBENCH_MVT_BODY6 \
  x2[i] += dot;


#define POLYBENCH_MVT_BODY1_RAJA \
  dot = 0.0;

#define POLYBENCH_MVT_BODY2_RAJA \
  dot += Aview(i, j) * y1view(j); 

#define POLYBENCH_MVT_BODY3_RAJA \
  x1view(i) += dot;

#define POLYBENCH_MVT_BODY4_RAJA \
  dot = 0.0;

#define POLYBENCH_MVT_BODY5_RAJA \
  dot += Aview(j, i) * y2view(i); 

#define POLYBENCH_MVT_BODY6_RAJA \
  x2view(i) += dot;


#define POLYBENCH_MVT_VIEWS_RAJA \
  using VIEW_1 = RAJA::View<Real_type, \
                            RAJA::Layout<1, Index_type, 0>>; \
\
  using VIEW_2 = RAJA::View<Real_type, \
                            RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_1 x1view(x1, RAJA::Layout<1>(N)); \
  VIEW_1 x2view(x2, RAJA::Layout<1>(N)); \
  VIEW_1 y1view(y1, RAJA::Layout<1>(N)); \
  VIEW_1 y2view(y2, RAJA::Layout<1>(N)); \
  VIEW_2 Aview(A, RAJA::Layout<2>(N, N));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_MVT : public KernelBase
{
public:

  POLYBENCH_MVT(const RunParams& params);

  ~POLYBENCH_MVT();


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
  Real_ptr m_x1;
  Real_ptr m_x2;
  Real_ptr m_y1;
  Real_ptr m_y2;
  Real_ptr m_A;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
