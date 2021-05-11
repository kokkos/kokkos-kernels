//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SORTPAIRS kernel reference implementation:
///
/// std::sort(x+ibegin, x+iend);
///

#ifndef RAJAPerf_Algorithm_SORTPAIRS_HPP
#define RAJAPerf_Algorithm_SORTPAIRS_HPP

#define SORTPAIRS_DATA_SETUP \
  Real_ptr x = m_x;          \
  Real_ptr i = m_i;

#define SORTPAIRS_RAJA_ARGS  \
  x + iend*irep + ibegin, x + iend*irep + iend, i + iend*irep + ibegin


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace algorithm
{

class SORTPAIRS : public KernelBase
{
public:

  SORTPAIRS(const RunParams& params);

  ~SORTPAIRS();

  void setUp(VariantID vid);
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runHipVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid)
  {
    std::cout << "\n  SORTPAIRS : Unknown OMP Target variant id = " << vid << std::endl;
  }

private:
  Real_ptr m_x;
  Real_ptr m_i;
};

} // end namespace algorithm
} // end namespace rajaperf

#endif // closing endif for header file include guard
