//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// NESTED_INIT kernel reference implementation:
///
/// for (Index_type k = 0; k < nk; ++k ) {
///   for (Index_type j = 0; j < nj; ++j ) {
///     for (Index_type i = 0; i < ni; ++i ) {
///       array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
///     }
///   }
/// }
///

#ifndef RAJAPerf_Basic_NESTED_INIT_HPP
#define RAJAPerf_Basic_NESTED_INIT_HPP


#define NESTED_INIT_DATA_SETUP \
  Real_ptr array = m_array; \
  Index_type ni = m_ni; \
  Index_type nj = m_nj; \
  Index_type nk = m_nk;

#define NESTED_INIT_BODY  \
  array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class NESTED_INIT : public KernelBase
{
public:

  NESTED_INIT(const RunParams& params);

  ~NESTED_INIT();

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
  Index_type m_array_length;

  Real_ptr m_array;

  Index_type m_ni;
  Index_type m_nj;
  Index_type m_nk;
  Index_type m_nk_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
