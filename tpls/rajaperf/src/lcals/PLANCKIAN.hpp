//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// PLANCKIAN kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] = u[i] / v[i];
///   w[i] = x[i] / ( exp( y[i] ) - 1.0 );
/// }
///

#ifndef RAJAPerf_Lcals_PLANCKIAN_HPP
#define RAJAPerf_Lcals_PLANCKIAN_HPP


#define PLANCKIAN_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr u = m_u; \
  Real_ptr v = m_v; \
  Real_ptr w = m_w;

#define PLANCKIAN_BODY  \
  y[i] = u[i] / v[i]; \
  w[i] = x[i] / ( exp( y[i] ) - 1.0 );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class PLANCKIAN : public KernelBase
{
public:

  PLANCKIAN(const RunParams& params);

  ~PLANCKIAN();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_u;
  Real_ptr m_v;
  Real_ptr m_w;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
