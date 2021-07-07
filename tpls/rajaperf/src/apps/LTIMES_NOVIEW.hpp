//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// LTIMES_NOVIEW kernel reference implementation:
///
/// for (Index_type z = 0; z < num_z; ++z ) {
///   for (Index_type g = 0; g < num_g; ++g ) {
///     for (Index_type m = 0; z < num_m; ++m ) {
///       for (Index_type d = 0; d < num_d; ++d ) {
///
///         phi[m+ (g * num_m) + (z * num_m * num_g)] +=
///           ell[d+ (m * num_d)] * psi[d+ (g * num_d) + (z * num_d * num_g];
///
///       }
///     }
///   }
/// }
///

#ifndef RAJAPerf_Apps_LTIMES_NOVIEW_HPP
#define RAJAPerf_Apps_LTIMES_NOVIEW_HPP

#define LTIMES_NOVIEW_DATA_SETUP \
  Real_ptr phidat = m_phidat; \
  Real_ptr elldat = m_elldat; \
  Real_ptr psidat = m_psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m;

#define LTIMES_NOVIEW_BODY \
  phidat[m+ (g * num_m) + (z * num_m * num_g)] += \
    elldat[d+ (m * num_d)] * psidat[d+ (g * num_d) + (z * num_d * num_g)];

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace apps
{

class LTIMES_NOVIEW : public KernelBase
{
public:

  LTIMES_NOVIEW(const RunParams& params);

  ~LTIMES_NOVIEW();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_phidat;
  Real_ptr m_elldat;
  Real_ptr m_psidat;

  Index_type m_num_d_default; 
  Index_type m_num_z_default; 
  Index_type m_num_g_default; 
  Index_type m_num_m_default; 

  Index_type m_num_d; 
  Index_type m_num_z; 
  Index_type m_num_g; 
  Index_type m_num_m; 

  Index_type m_philen;
  Index_type m_elllen;
  Index_type m_psilen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
