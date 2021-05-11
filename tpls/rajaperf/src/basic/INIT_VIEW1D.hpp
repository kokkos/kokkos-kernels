//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT_VIEW1D kernel reference implementation:
///
/// const Real_type val = ...;
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = (i+1) * val;
/// }
///
/// RAJA variants use a "View" and "Layout" to do the same thing. These 
/// RAJA constructs provide little benfit in 1D, but they are used here
/// to exercise those RAJA mechanics in the simplest scenario.
///

#ifndef RAJAPerf_Basic_INIT_VIEW1D_HPP
#define RAJAPerf_Basic_INIT_VIEW1D_HPP


#define INIT_VIEW1D_DATA_SETUP \
  Real_ptr a = m_a; \
  const Real_type v = m_val;

#define INIT_VIEW1D_BODY  \
  a[i] = (i+1) * v;

#define INIT_VIEW1D_BODY_RAJA  \
  view(i) = (i+1) * v;

#define INIT_VIEW1D_VIEW_RAJA \
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1, Index_type, 0> >; \
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class INIT_VIEW1D : public KernelBase
{
public:

  INIT_VIEW1D(const RunParams& params);

  ~INIT_VIEW1D();

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
  Real_type m_val;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
