//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
  setDefaultSize(100000);
  setDefaultReps(4000);

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

INT_PREDICT::~INT_PREDICT() 
{
}

void INT_PREDICT::setUp(VariantID vid)
{
  m_array_length = getRunSize() * 13;
  m_offset = getRunSize();

  m_px_initval = 1.0;
  allocAndInitDataConst(m_px, m_array_length, m_px_initval, vid);

  initData(m_dm22);
  initData(m_dm23);
  initData(m_dm24);
  initData(m_dm25);
  initData(m_dm26);
  initData(m_dm27);
  initData(m_dm28);
  initData(m_c0);
}

void INT_PREDICT::updateChecksum(VariantID vid)
{
  for (Index_type i = 0; i < getRunSize(); ++i) {
    m_px[i] -= m_px_initval;
  }
  
  checksum[vid] += calcChecksum(m_px, getRunSize());
}

void INT_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
}

} // end namespace lcals
} // end namespace rajaperf
