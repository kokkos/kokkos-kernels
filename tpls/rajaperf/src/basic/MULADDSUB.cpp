//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


MULADDSUB::MULADDSUB(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULADDSUB, params)
{
  setDefaultSize(100000);
  setDefaultReps(3500);

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

  setVariantDefined( Kokkos_Lambda );




}

MULADDSUB::~MULADDSUB()
{
}

void MULADDSUB::setUp(VariantID vid)
{
  allocAndInitDataConst(m_out1, getRunSize(), 0.0, vid);
  allocAndInitDataConst(m_out2, getRunSize(), 0.0, vid);
  allocAndInitDataConst(m_out3, getRunSize(), 0.0, vid);
  allocAndInitData(m_in1, getRunSize(), vid);
  allocAndInitData(m_in2, getRunSize(), vid);
}

void MULADDSUB::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunSize());
  checksum[vid] += calcChecksum(m_out2, getRunSize());
  checksum[vid] += calcChecksum(m_out3, getRunSize());
}

void MULADDSUB::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_out1);
  deallocData(m_out2);
  deallocData(m_out3);
  deallocData(m_in1);
  deallocData(m_in2);
}

} // end namespace basic
} // end namespace rajaperf
