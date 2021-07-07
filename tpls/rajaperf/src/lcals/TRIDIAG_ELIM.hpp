//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRIDIAG_ELIM kernel reference implementation:
///
/// Note: kernel is altered to enable parallelism (original did not have
///       separate input and output arrays for 'x').
///
/// for (Index_type i = 1; i < N; ++i ) {
///   xout[i] = z[i] * ( y[i] - xin[i-1] );
/// }
///

#ifndef RAJAPerf_Lcals_TRIDIAG_ELIM_HPP
#define RAJAPerf_Lcals_TRIDIAG_ELIM_HPP


#define TRIDIAG_ELIM_DATA_SETUP \
  Real_ptr xout = m_xout; \
  Real_ptr xin = m_xin; \
  Real_ptr y = m_y; \
  Real_ptr z = m_z;

#define TRIDIAG_ELIM_BODY  \
  xout[i] = z[i] * ( y[i] - xin[i-1] );


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class TRIDIAG_ELIM : public KernelBase
{
public:

  TRIDIAG_ELIM(const RunParams& params);

  ~TRIDIAG_ELIM();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_xout;
  Real_ptr m_xin;
  Real_ptr m_y;
  Real_ptr m_z;

  Index_type m_N;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
