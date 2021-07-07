//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// REDUCE3_INT kernel reference implementation:
///
/// Int_type vsum = m_vsum_init;
/// Int_type vmin = m_vmin_init;
/// Int_type vmax = m_vmax_init;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   vsum += vec[i] ;
///   vmin = RAJA_MIN(vmin, vec[i]) ;
///   vmax = RAJA_MAX(vmax, vec[i]) ;
/// }
///
/// m_vsum += vsum;
/// m_vmin = RAJA_MIN(m_vmin, vmin);
/// m_vmax = RAJA_MAX(m_vmax, vmax);
///
/// RAJA_MIN/MAX are macros that do what you would expect.
///

#ifndef RAJAPerf_Basic_REDUCE3_INT_HPP
#define RAJAPerf_Basic_REDUCE3_INT_HPP


#define REDUCE3_INT_DATA_SETUP \
  Int_ptr vec = m_vec; \

#define REDUCE3_INT_BODY  \
  vsum += vec[i] ; \
  vmin = RAJA_MIN(vmin, vec[i]) ; \
  vmax = RAJA_MAX(vmax, vec[i]) ;

#define REDUCE3_INT_BODY_RAJA  \
  vsum += vec[i] ; \
  vmin.min(vec[i]) ; \
  vmax.max(vec[i]) ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class REDUCE3_INT : public KernelBase
{
public:

  REDUCE3_INT(const RunParams& params);

  ~REDUCE3_INT();

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
  Int_ptr m_vec;
  Int_type m_vsum;
  Int_type m_vsum_init;
  Int_type m_vmax;
  Int_type m_vmax_init;
  Int_type m_vmin;
  Int_type m_vmin_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
