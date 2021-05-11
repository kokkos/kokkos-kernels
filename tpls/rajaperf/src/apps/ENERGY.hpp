//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ENERGY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] = e_old[i] - 0.5 * delvc[i] *
///              (p_old[i] + q_old[i]) + 0.5 * work[i];
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if ( delvc[i] > 0.0 ) {
///      q_new[i] = 0.0 ;
///   }
///   else {
///      Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ;
///      Real_type ssc = ( pbvc[i] * e_new[i]
///         + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
///   }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] = e_new[i] + 0.5 * delvc[i]
///              * ( 3.0*(p_old[i] + q_old[i])
///                  - 4.0*(pHalfStep[i] + q_new[i])) ;
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   e_new[i] += 0.5 * work[i];
///   if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; }
///   if ( e_new[i]  < emin ) { e_new[i] = emin ; }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type q_tilde ;
///   if (delvc[i] > 0.0) {
///      q_tilde = 0. ;
///   }
///   else {
///      Real_type ssc = ( pbvc[i] * e_new[i]
///          + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
///   }
///   e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i])
///                          - 8.0*(pHalfStep[i] + q_new[i])
///                          + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ;
///   if ( fabs(e_new[i]) < e_cut ) {
///      e_new[i] = 0.0  ;
///   }
///   if ( e_new[i]  < emin ) {
///      e_new[i] = emin ;
///   }
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   if ( delvc[i] <= 0.0 ) {
///      Real_type ssc = ( pbvc[i] * e_new[i]
///              + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ;
///      if ( ssc <= 0.1111111e-36 ) {
///         ssc = 0.3333333e-18 ;
///      } else {
///         ssc = sqrt(ssc) ;
///      }
///      q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
///      if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ;
///   }
/// }
///

#ifndef RAJAPerf_Apps_ENERGY_HPP
#define RAJAPerf_Apps_ENERGY_HPP

#define ENERGY_DATA_SETUP \
  Real_ptr e_new = m_e_new; \
  Real_ptr e_old = m_e_old; \
  Real_ptr delvc = m_delvc; \
  Real_ptr p_new = m_p_new; \
  Real_ptr p_old = m_p_old; \
  Real_ptr q_new = m_q_new; \
  Real_ptr q_old = m_q_old; \
  Real_ptr work = m_work; \
  Real_ptr compHalfStep = m_compHalfStep; \
  Real_ptr pHalfStep = m_pHalfStep; \
  Real_ptr bvc = m_bvc; \
  Real_ptr pbvc = m_pbvc; \
  Real_ptr ql_old = m_ql_old; \
  Real_ptr qq_old = m_qq_old; \
  Real_ptr vnewc = m_vnewc; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut;

#define ENERGY_BODY1 \
  e_new[i] = e_old[i] - 0.5 * delvc[i] * \
             (p_old[i] + q_old[i]) + 0.5 * work[i];

#define ENERGY_BODY2 \
  if ( delvc[i] > 0.0 ) { \
     q_new[i] = 0.0 ; \
  } \
  else { \
     Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]) ; \
     Real_type ssc = ( pbvc[i] * e_new[i] \
        + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
  }

#define ENERGY_BODY3 \
  e_new[i] = e_new[i] + 0.5 * delvc[i] \
             * ( 3.0*(p_old[i] + q_old[i]) \
                 - 4.0*(pHalfStep[i] + q_new[i])) ;

#define ENERGY_BODY4 \
  e_new[i] += 0.5 * work[i]; \
  if ( fabs(e_new[i]) < e_cut ) { e_new[i] = 0.0  ; } \
  if ( e_new[i]  < emin ) { e_new[i] = emin ; }

#define ENERGY_BODY5 \
  Real_type q_tilde ; \
  if (delvc[i] > 0.0) { \
     q_tilde = 0. ; \
  } \
  else { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
         + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_tilde = (ssc*ql_old[i] + qq_old[i]) ; \
  } \
  e_new[i] = e_new[i] - ( 7.0*(p_old[i] + q_old[i]) \
                         - 8.0*(pHalfStep[i] + q_new[i]) \
                         + (p_new[i] + q_tilde)) * delvc[i] / 6.0 ; \
  if ( fabs(e_new[i]) < e_cut ) { \
     e_new[i] = 0.0  ; \
  } \
  if ( e_new[i]  < emin ) { \
     e_new[i] = emin ; \
  }

#define ENERGY_BODY6 \
  if ( delvc[i] <= 0.0 ) { \
     Real_type ssc = ( pbvc[i] * e_new[i] \
             + vnewc[i] * vnewc[i] * bvc[i] * p_new[i] ) / rho0 ; \
     if ( ssc <= 0.1111111e-36 ) { \
        ssc = 0.3333333e-18 ; \
     } else { \
        ssc = sqrt(ssc) ; \
     } \
     q_new[i] = (ssc*ql_old[i] + qq_old[i]) ; \
     if (fabs(q_new[i]) < q_cut) q_new[i] = 0.0 ; \
  }


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class ENERGY : public KernelBase
{
public:

  ENERGY(const RunParams& params);

  ~ENERGY();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_e_new;
  Real_ptr m_e_old;
  Real_ptr m_delvc;
  Real_ptr m_p_new;
  Real_ptr m_p_old; 
  Real_ptr m_q_new; 
  Real_ptr m_q_old; 
  Real_ptr m_work; 
  Real_ptr m_compHalfStep; 
  Real_ptr m_pHalfStep; 
  Real_ptr m_bvc; 
  Real_ptr m_pbvc; 
  Real_ptr m_ql_old; 
  Real_ptr m_qq_old; 
  Real_ptr m_vnewc; 

  Real_type m_rho0;
  Real_type m_e_cut;
  Real_type m_emin;
  Real_type m_q_cut;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
