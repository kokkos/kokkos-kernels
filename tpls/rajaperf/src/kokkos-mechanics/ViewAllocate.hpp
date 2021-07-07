//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ViewAllocate kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] += a * x[i] ;
/// }
///

#ifndef RAJAPerf_Basic_ViewAllocate_HPP
#define RAJAPerf_Basic_ViewAllocate_HPP

#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace kokkos_mechanics
{

class ViewAllocate : public KernelBase
{
public:

  ViewAllocate(const RunParams& params);

  ~ViewAllocate();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  void runKokkosSeqVariant(VariantID vid);
  void runKokkosOpenMPVariant(VariantID vid);
  void runKokkosCudaVariant(VariantID vid);
  void runKokkosOpenMPTargetVariant(VariantID vid);
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
