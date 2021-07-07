//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DOT kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   dot += a[i] * b[i];
/// }
///

#ifndef RAJAPerf_Stream_DOT_HPP
#define RAJAPerf_Stream_DOT_HPP

#define DOT_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr b = m_b;

#define DOT_BODY  \
  dot += a[i] * b[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class DOT : public KernelBase
{
public:

  DOT(const RunParams& params);

  ~DOT();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_type m_dot;
  Real_type m_dot_init;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
