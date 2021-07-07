//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "SORTPAIRS.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


SORTPAIRS::SORTPAIRS(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_SORTPAIRS, params)
{
   setDefaultSize(100000);
   setDefaultReps(50);

  setVariantDefined( Base_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( RAJA_CUDA );

  setVariantDefined( RAJA_HIP );
}

SORTPAIRS::~SORTPAIRS()
{
}

void SORTPAIRS::setUp(VariantID vid)
{
  allocAndInitDataRandValue(m_x, getRunSize()*getRunReps(), vid);
  allocAndInitDataRandValue(m_i, getRunSize()*getRunReps(), vid);
}

void SORTPAIRS::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize()*getRunReps());
  checksum[vid] += calcChecksum(m_i, getRunSize()*getRunReps());
}

void SORTPAIRS::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_i);
}

} // end namespace algorithm
} // end namespace rajaperf
