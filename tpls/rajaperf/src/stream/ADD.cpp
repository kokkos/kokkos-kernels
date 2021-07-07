//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


ADD::ADD(const RunParams& params)
  : KernelBase(rajaperf::Stream_ADD, params)
{
  setDefaultSize(1000000);
  setDefaultReps(1000);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );
}

ADD::~ADD()
{
}

void ADD::setUp(VariantID vid)
{
  allocAndInitData(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitDataConst(m_c, getRunSize(), 0.0, vid);
}

void ADD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_c, getRunSize());
}

void ADD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
