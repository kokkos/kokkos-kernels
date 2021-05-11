//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRAP_INT kernel reference implementation:
///
/// Real_type trap_int_func(Real_type x,
///                         Real_type y,
///                         Real_type xp,
///                         Real_type yp)
/// {
///    Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
///    denom = 1.0/sqrt(denom);
///    return denom;
/// }
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///    Real_type x = x0 + i*h;
///    sumx += trap_int_func(x, y, xp, yp);
/// }
///

#ifndef RAJAPerf_Basic_TRAP_INT_HPP
#define RAJAPerf_Basic_TRAP_INT_HPP


#define TRAP_INT_DATA_SETUP \
  Real_type x0 = m_x0; \
  Real_type xp = m_xp; \
  Real_type y = m_y; \
  Real_type yp = m_yp; \
  Real_type h = m_h;

#define TRAP_INT_BODY \
  Real_type x = x0 + i*h; \
  sumx += trap_int_func(x, y, xp, yp);


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class TRAP_INT : public KernelBase
{
public:

  TRAP_INT(const RunParams& params);

  ~TRAP_INT();

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
  Real_type m_x0;
  Real_type m_xp;
  Real_type m_y;
  Real_type m_yp;
  Real_type m_h;
  Real_type m_sumx_init;

  Real_type m_sumx;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
