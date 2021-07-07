//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ATOMIC_PI kernel reference implementation:
///
/// const int N = ...;  -- num [0, 1] sub-intervals used in Riemann integration
/// const double dx = 1.0 / double(num_bins);
///
/// double pi = 0.0;
/// for (Index_type i = 0; i < N; ++i ) {
///   double x = (double(i) + 0.5) * dx;
///   pi += dx / (1.0 + x * x);
/// }
/// pi *= 4.0;
///

#ifndef RAJAPerf_Basic_ATOMIC_PI_HPP
#define RAJAPerf_Basic_ATOMIC_PI_HPP

#define ATOMIC_PI_DATA_SETUP \
  Real_type dx = m_dx; \
  Real_ptr pi = m_pi;

#define ATOMIC_PI_FUNCTOR_CONSTRUCT \
  dx(m_dx), \
  pi(m_pi)

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class ATOMIC_PI : public KernelBase
{
public:

  ATOMIC_PI(const RunParams& params);

  ~ATOMIC_PI();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
#ifdef RUN_KOKKOS
  void runKokkosVariant(VariantID vid);
  
  
  
#endif

private:
  Real_type m_dx;
  Real_ptr m_pi;
  Real_type m_pi_init;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
