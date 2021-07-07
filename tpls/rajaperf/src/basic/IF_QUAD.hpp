//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// IF_QUAD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type s = b[i]*b[i] - 4.0*a[i]*c[i];
///   if ( s >= 0 ) {
///     s = sqrt(s);
///     x2[i] = (-b[i]+s)/(2.0*a[i]);
///     x1[i] = (-b[i]-s)/(2.0*a[i]);
///   } else {
///     x2[i] = 0.0;
///     x1[i] = 0.0;
///   }
/// }
///

#ifndef RAJAPerf_Basic_IF_QUAD_HPP
#define RAJAPerf_Basic_IF_QUAD_HPP

#define IF_QUAD_DATA_SETUP \
  Real_ptr a = m_a; \
  Real_ptr b = m_b; \
  Real_ptr c = m_c; \
  Real_ptr x1 = m_x1; \
  Real_ptr x2 = m_x2;

#define IF_QUAD_BODY  \
  Real_type s = b[i]*b[i] - 4.0*a[i]*c[i]; \
  if ( s >= 0 ) { \
    s = sqrt(s); \
    x2[i] = (-b[i]+s)/(2.0*a[i]); \
    x1[i] = (-b[i]-s)/(2.0*a[i]); \
  } else { \
    x2[i] = 0.0; \
    x1[i] = 0.0; \
  }

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class IF_QUAD : public KernelBase
{
public:

  IF_QUAD(const RunParams& params);

  ~IF_QUAD();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  void runKokkosVariant(VariantID vid);
  
  
  

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_ptr m_x1;
  Real_ptr m_x2;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
