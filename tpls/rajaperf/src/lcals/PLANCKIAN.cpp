//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


PLANCKIAN::PLANCKIAN(const RunParams& params)
  : KernelBase(rajaperf::Lcals_PLANCKIAN, params)
{
  setDefaultSize(100000);
  setDefaultReps(460);

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

PLANCKIAN::~PLANCKIAN() 
{
}

void PLANCKIAN::setUp(VariantID vid)
{
  allocAndInitData(m_x, getRunSize(), vid);
  allocAndInitData(m_y, getRunSize(), vid);
  allocAndInitData(m_u, getRunSize(), vid);
  allocAndInitData(m_v, getRunSize(), vid);
  allocAndInitDataConst(m_w, getRunSize(), 0.0, vid);
}

void PLANCKIAN::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, getRunSize());
}

void PLANCKIAN::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_u);
  deallocData(m_v);
  deallocData(m_w);
}

} // end namespace lcals
} // end namespace rajaperf
