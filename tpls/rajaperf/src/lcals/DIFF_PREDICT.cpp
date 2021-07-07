//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


DIFF_PREDICT::DIFF_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_DIFF_PREDICT, params)
{
  setDefaultSize(100000);
  setDefaultReps(2000);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );
                     
  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );
  
  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );
      
  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );
        
  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

DIFF_PREDICT::~DIFF_PREDICT() 
{
}

void DIFF_PREDICT::setUp(VariantID vid)
{
  m_array_length = getRunSize() * 14;
  m_offset = getRunSize();

  allocAndInitDataConst(m_px, m_array_length, 0.0, vid);
  allocAndInitData(m_cx, m_array_length, vid);
}

void DIFF_PREDICT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_px, m_array_length);
}

void DIFF_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
  deallocData(m_cx);
}

} // end namespace lcals
} // end namespace rajaperf
