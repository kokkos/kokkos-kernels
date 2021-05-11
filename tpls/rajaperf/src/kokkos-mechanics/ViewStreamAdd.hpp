//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ViewStreamAdd kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] += a * x[i] ;
/// }
///

#ifndef RAJAPerf_Basic_ViewStreamAdd_HPP
#define RAJAPerf_Basic_ViewStreamAdd_HPP

#define ViewStreamAdd_DATA_SETUP \
  Real_ptr x = m_x; \
  Real_ptr y = m_y; \
  Real_type a = m_a;

#define ViewStreamAdd_FUNCTOR_CONSTRUCT \
  x(m_x),\
  y(m_y), \
  a(m_a)

#define ViewStreamAdd_BODY  \
  y[i] += a * x[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace kokkos_mechanics
{

class ViewStreamAdd : public KernelBase
{
public:

  ViewStreamAdd(const RunParams& params);

  ~ViewStreamAdd();

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
private:
  using VT=Kokkos::View<float*, Kokkos::HostSpace>;
  VT h_a;
  VT h_b;
  VT h_c;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
