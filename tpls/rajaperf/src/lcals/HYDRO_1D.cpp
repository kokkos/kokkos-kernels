//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


HYDRO_1D::HYDRO_1D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_1D, params)
{
  setDefaultSize(100000);
  setDefaultReps(12500);

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

HYDRO_1D::~HYDRO_1D() 
{
}

void HYDRO_1D::setUp(VariantID vid)
{
  m_array_length = getRunSize() + 12;

  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitData(m_y, m_array_length, vid);
  allocAndInitData(m_z, m_array_length, vid);

  initData(m_q, vid);
  initData(m_r, vid);
  initData(m_t, vid);
}

void HYDRO_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getRunSize());
}

void HYDRO_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace lcals
} // end namespace rajaperf
