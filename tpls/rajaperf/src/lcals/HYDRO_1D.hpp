//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// HYDRO_1D kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );
/// }
///

#ifndef RAJAPerf_Lcals_HYDRO_1D_HPP
#define RAJAPerf_Lcals_HYDRO_1D_HPP


#define HYDRO_1D_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z; \
\
  const Real_type q = m_q; \
  const Real_type r = m_r; \
  const Real_type t = m_t;

#define HYDRO_1D_BODY  \
  x[i] = q + y[i]*( r*z[i+10] + t*z[i+11] );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class HYDRO_1D : public KernelBase
{
public:

  HYDRO_1D(const RunParams& params);

  ~HYDRO_1D();

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
  Real_ptr m_z;

  Real_type m_q;
  Real_type m_r;
  Real_type m_t;

  Index_type m_array_length; 
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
