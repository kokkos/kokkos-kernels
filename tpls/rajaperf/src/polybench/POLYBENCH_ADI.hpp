//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// POLYBENCH_ADI kernel reference implementation:
///
///  DX = 1.0/N;
///  DY = 1.0/N;
///  DT = 1.0/TSTEPS;
///  B1 = 2.0;
///  B2 = 1.0;
///  mul1 = B1 * DT / (DX * DX);
///  mul2 = B2 * DT / (DY * DY);
///
///  a = -mul1 / 2.0;
///  b = 1.0 + mul1;
///  c = a;
///  d = -mul2 / 2.0;
///  e = 1.0 + mul2;
///  f = d;
///
/// for (t=1; t<=TSTEPS; t++) {
///    //Column Sweep
///    for (i=1; i<N-1; i++) {
///      v[0][i] = 1.0;
///      p[i][0] = 0.0;
///      q[i][0] = v[0][i];
///      for (j=1; j<N-1; j++) {
///        p[i][j] = -c / (a*p[i][j-1]+b);
///        q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - 
///                   f*u[j][i+1]-a*q[i][j-1]) / (a*p[i][j-1]+b);
///      }
///      
///      v[N-1][i] = 1.0;
///      for (k=N-2; k>=1; k--) {
///        v[k][i] = p[i][k] * v[k+1][i] + q[i][k];
///      }
///    }
///    //Row Sweep
///    for (i=1; i<N-1; i++) {
///      u[i][0] = 1.0;
///      p[i][0] = 0.0;
///      q[i][0] = u[i][0];
///      for (j=1; j<N-1; j++) {
///        p[i][j] = -f / (d*p[i][j-1]+e);
///        q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - 
///                  c*v[i+1][j]-d*q[i][j-1]) / (d*p[i][j-1]+e);
///      }
///      u[i][N-1] = 1.0;
///      for (k=N-2; k>=1; k--) {
///        u[i][k] = p[i][k] * u[i][k+1] + q[i][k];
///      }
///    }
///  }



#ifndef RAJAPerf_POLYBENCH_ADI_HPP
#define RAJAPerf_POLYBENCH_ADI_HPP


#define POLYBENCH_ADI_DATA_SETUP \
  const Index_type n = m_n; \
  const Index_type tsteps = m_tsteps; \
\
  Real_type DX = 1.0/(Real_type)n; \
  Real_type DY = 1.0/(Real_type)n; \
  Real_type DT = 1.0/(Real_type)tsteps; \
  Real_type B1 = 2.0; \
  Real_type B2 = 1.0; \
  Real_type mul1 = B1 * DT / (DX * DX); \
  Real_type mul2 = B2 * DT / (DY * DY); \
  Real_type a = -mul1 / 2.0; \
  Real_type b = 1.0 + mul1; \
  Real_type c = a; \
  Real_type d = -mul2 /2.0; \
  Real_type e = 1.0 + mul2; \
  Real_type f = d; \
\
  Real_ptr U = m_U; \
  Real_ptr V = m_V; \
  Real_ptr P = m_P; \
  Real_ptr Q = m_Q;


#define POLYBENCH_ADI_BODY2 \
  V[0 * n + i] = 1.0; \
  P[i * n + 0] = 0.0; \
  Q[i * n + 0] = V[0 * n + i];

#define POLYBENCH_ADI_BODY3 \
  P[i * n + j] = -c / (a * P[i * n + j-1] + b); \
  Q[i * n + j] = (-d * U[j * n + i-1] + (1.0 + 2.0*d) * U[j * n + i] - \
                 f * U[j * n + i + 1] - a * Q[i * n + j-1]) / \
                    (a * P[i * n + j-1] + b); 

#define POLYBENCH_ADI_BODY4 \
  V[(n-1) * n + i] = 1.0;

#define POLYBENCH_ADI_BODY5 \
  V[k * n + i]  = P[i * n + k] * V[(k+1) * n + i] + Q[i * n + k]; 

#define POLYBENCH_ADI_BODY6 \
  U[i * n + 0] = 1.0; \
  P[i * n + 0] = 0.0; \
  Q[i * n + 0] = U[i * n + 0];

#define POLYBENCH_ADI_BODY7 \
  P[i * n + j] = -f / (d * P[i * n + j-1] + e); \
  Q[i * n + j] = (-a * V[(i-1) * n + j] + (1.0 + 2.0*a) * V[i * n + j] - \
                 c * V[(i + 1) * n + j] - d * Q[i * n + j-1]) / \
                    (d * P[i * n + j-1] + e);

#define POLYBENCH_ADI_BODY8 \
  U[i * n + n-1] = 1.0;

#define POLYBENCH_ADI_BODY9 \
  U[i * n + k] = P[i * n + k] * U[i * n + k +1] + Q[i * n + k]; 


#define POLYBENCH_ADI_BODY2_RAJA \
  Vview(0, i) = 1.0; \
  Pview(i, 0) = 0.0; \
  Qview(i, 0) = Vview(0, i);

#define POLYBENCH_ADI_BODY3_RAJA \
  Pview(i, j) = -c / (a * Pview(i, j-1) + b); \
  Qview(i, j) = (-d * Uview(j, i-1) + (1.0 + 2.0*d) * Uview(j, i) - \
                 f * Uview(j, i+1) - a * Qview(i, j-1)) / \
                   (a * Pview(i, j-1) + b);

#define POLYBENCH_ADI_BODY4_RAJA \
  Vview(n-1, i) = 1.0;

#define POLYBENCH_ADI_BODY5_RAJA \
  Vview(k, i)  = Pview(i, k) * Vview(k+1, i) + Qview(i, k);

#define POLYBENCH_ADI_BODY6_RAJA \
  Uview(i, 0) = 1.0; \
  Pview(i, 0) = 0.0; \
  Qview(i, 0) = Uview(i, 0);

#define POLYBENCH_ADI_BODY7_RAJA \
  Pview(i, j) = -f / (d * Pview(i, j-1) + e); \
  Qview(i, j) = (-a * Vview(i-1, j) + (1.0 + 2.0*a) * Vview(i, j) - \
                 c * Vview(i + 1, j) - d * Qview(i, j-1)) / \
                   (d * Pview(i, j-1) + e);

#define POLYBENCH_ADI_BODY8_RAJA \
  Uview(i, n-1) = 1.0;

#define POLYBENCH_ADI_BODY9_RAJA \
  Uview(i, k) = Pview(i, k) * Uview(i, k+1) + Qview(i, k);


#define POLYBENCH_ADI_VIEWS_RAJA \
  using VIEW_TYPE = RAJA::View<Real_type, \
                          RAJA::Layout<2, Index_type, 1>>; \
\
  VIEW_TYPE Uview(U, RAJA::Layout<2>(n, n)); \
  VIEW_TYPE Vview(V, RAJA::Layout<2>(n, n)); \
  VIEW_TYPE Pview(P, RAJA::Layout<2>(n, n)); \
  VIEW_TYPE Qview(Q, RAJA::Layout<2>(n, n));


#include "common/KernelBase.hpp"

namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_ADI : public KernelBase
{
public:

  POLYBENCH_ADI(const RunParams& params);

  ~POLYBENCH_ADI();

 
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
  Index_type m_tsteps;

  Real_ptr m_U;
  Real_ptr m_V;
  Real_ptr m_P;
  Real_ptr m_Q;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
