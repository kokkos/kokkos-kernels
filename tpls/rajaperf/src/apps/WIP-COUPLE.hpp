//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// COUPLE kernel reference implementation:
///
/// for (Index_type k = kmin ; k < kmax ; ++k ) {
///   for (Index_type j = jmin; j < jmax; j++) {
///
///      Index_type it0=    ((k)*(jmax+1) + (j))*(imax+1) ;
///      Index_type idenac= ((k)*(jmax+2) + (j))*(imax+2) ;
///
///      for (Index_type i = imin; i < imax; i++) {
///
///         Complex_type c1 = c10 * denac[idenac+i];
///         Complex_type c2 = c20 * denlw[it0+i];
///
///         /* promote to doubles to avoid possible divide by zero */
///         Real_type c1re = real(c1);  Real_type c1im = imag(c1);
///         Real_type c2re = real(c2);  Real_type c2im = imag(c2);
///
///         /* lamda = sqrt(|c1|^2 + |c2|^2) uses doubles to avoid underflow. */
///         Real_type zlam = c1re*c1re + c1im*c1im +
///                          c2re*c2re + c2im*c2im + 1.0e-34;
///         zlam = sqrt(zlam);
///         Real_type snlamt = sin(zlam * dt * 0.5);
///         Real_type cslamt = cos(zlam * dt * 0.5);
///
///         Complex_type a0t = t0[it0+i];
///         Complex_type a1t = t1[it0+i];
///         Complex_type a2t = t2[it0+i] * fratio;
///
///         Real_type r_zlam= 1.0/zlam;
///         c1 *= r_zlam;
///         c2 *= r_zlam;
///         Real_type zac1 = zabs2(c1);
///         Real_type zac2 = zabs2(c2);
///
///         /* compute new A0 */
///         Complex_type z3 = ( c1 * a1t + c2 * a2t ) * snlamt ;
///         t0[it0+i] = a0t * cslamt -  ireal * z3;
///
///         /* compute new A1  */
///         Real_type r = zac1 * cslamt + zac2;
///         Complex_type z5 = c2 * a2t;
///         Complex_type z4 = conj(c1) * z5 * (cslamt-1);
///         z3 = conj(c1) * a0t * snlamt;
///         t1[it0+i] = a1t * r + z4 - ireal * z3;
///
///         /* compute new A2  */
///         r = zac1 + zac2 * cslamt;
///         z5 = c1 * a1t;
///         z4 = conj(c2) * z5 * (cslamt-1);
///         z3 = conj(c2) * a0t * snlamt;
///         t2[it0+i] = ( a2t * r + z4 - ireal * z3 ) * r_fratio;
///
///      } /* i loop */
///
///   } /* j loop */
/// } /* k loop */
///

#ifndef RAJAPerf_Apps_COUPLE_HPP
#define RAJAPerf_Apps_COUPLE_HPP

#define COUPLE_DATA_SETUP \
  Complex_ptr t0 = m_t0; \
  Complex_ptr t1 = m_t1; \
  Complex_ptr t2 = m_t2; \
  Complex_ptr denac = m_denac; \
  Complex_ptr denlw = m_denlw; \
  const Real_type dt = m_dt; \
  const Real_type c10 = m_c10; \
  const Real_type fratio = m_fratio; \
  const Real_type r_fratio = m_r_fratio; \
  const Real_type c20 = m_c20; \
  const Complex_type ireal = m_ireal; \
 \
  const Index_type imin = m_imin; \
  const Index_type imax = m_imax; \
  const Index_type jmin = m_jmin; \
  const Index_type jmax = m_jmax; \
  const Index_type kmin = m_kmin; \
  const Index_type kmax = m_kmax;

#define COUPLE_BODY \
for (Index_type j = jmin; j < jmax; j++) { \
 \
   Index_type it0=    ((k)*(jmax+1) + (j))*(imax+1) ; \
   Index_type idenac= ((k)*(jmax+2) + (j))*(imax+2) ; \
 \
   for (Index_type i = imin; i < imax; i++) { \
 \
      Complex_type c1 = c10 * denac[idenac+i]; \
      Complex_type c2 = c20 * denlw[it0+i]; \
 \
      /* promote to doubles to avoid possible divide by zero */ \
      Real_type c1re = real(c1);  Real_type c1im = imag(c1); \
      Real_type c2re = real(c2);  Real_type c2im = imag(c2); \
 \
      /* lamda = sqrt(|c1|^2 + |c2|^2) uses doubles to avoid underflow. */ \
      Real_type zlam = c1re*c1re + c1im*c1im + \
                       c2re*c2re + c2im*c2im + 1.0e-34; \
      zlam = sqrt(zlam); \
      Real_type snlamt = sin(zlam * dt * 0.5); \
      Real_type cslamt = cos(zlam * dt * 0.5); \
 \
      Complex_type a0t = t0[it0+i]; \
      Complex_type a1t = t1[it0+i]; \
      Complex_type a2t = t2[it0+i] * fratio; \
 \
      Real_type r_zlam= 1.0/zlam; \
      c1 *= r_zlam; \
      c2 *= r_zlam; \
      Real_type zac1 = zabs2(c1); \
      Real_type zac2 = zabs2(c2); \
 \
      /* compute new A0 */ \
      Complex_type z3 = ( c1 * a1t + c2 * a2t ) * snlamt ; \
      t0[it0+i] = a0t * cslamt -  ireal * z3; \
 \
      /* compute new A1  */ \
      Real_type r = zac1 * cslamt + zac2; \
      Complex_type z5 = c2 * a2t; \
      Complex_type z4 = conj(c1) * z5 * (cslamt-1); \
      z3 = conj(c1) * a0t * snlamt; \
      t1[it0+i] = a1t * r + z4 - ireal * z3; \
 \
      /* compute new A2  */ \
      r = zac1 + zac2 * cslamt; \
      z5 = c1 * a1t; \
      z4 = conj(c2) * z5 * (cslamt-1); \
      z3 = conj(c2) * a0t * snlamt; \
      t2[it0+i] = ( a2t * r + z4 - ireal * z3 ) * r_fratio; \
 \
   } /* i loop */ \
 \
} /* j loop */


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{
class ADomain;

class COUPLE : public KernelBase
{
public:

  COUPLE(const RunParams& params);

  ~COUPLE();

  Index_type getItsPerRep() const;

  void setUp(VariantID vid);
  void runKernel(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid) {(void) vid;}
  void runOpenMPVariant(VariantID vid) {(void) vid;}
  void runCudaVariant(VariantID vid) {(void) vid;}
  void runHipVariant(VariantID vid) {(void) vid;}
  void runOpenMPTargetVariant(VariantID vid) {(void) vid;}

private:
  Complex_ptr m_t0;
  Complex_ptr m_t1;
  Complex_ptr m_t2;
  Complex_ptr m_denac;
  Complex_ptr m_denlw;

  Real_type m_clight;
  Real_type m_csound;
  Real_type m_omega0;
  Real_type m_omegar;
  Real_type m_dt;
  Real_type m_c10;
  Real_type m_fratio;
  Real_type m_r_fratio;
  Real_type m_c20;
  Complex_type m_ireal;

  Index_type m_imin;
  Index_type m_imax;
  Index_type m_jmin;
  Index_type m_jmax;
  Index_type m_kmin;
  Index_type m_kmax;

  ADomain* m_domain;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
