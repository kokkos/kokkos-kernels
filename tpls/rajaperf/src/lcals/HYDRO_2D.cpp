//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{


HYDRO_2D::HYDRO_2D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_2D, params)
{
  m_jn = 1000;
  m_kn = 1000;

  m_s = 0.0041;
  m_t = 0.0037;

  setDefaultSize(m_jn);
  setDefaultReps(200);

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

HYDRO_2D::~HYDRO_2D() 
{
}

void HYDRO_2D::setUp(VariantID vid)
{
  m_kn = getRunSize();
  m_array_length = m_kn * m_jn;

  allocAndInitDataConst(m_zrout, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_zzout, m_array_length, 0.0, vid);
  allocAndInitData(m_za, m_array_length, vid);
  allocAndInitData(m_zb, m_array_length, vid);
  allocAndInitData(m_zm, m_array_length, vid);
  allocAndInitData(m_zp, m_array_length, vid);
  allocAndInitData(m_zq, m_array_length, vid);
  allocAndInitData(m_zr, m_array_length, vid);
  allocAndInitData(m_zu, m_array_length, vid);
  allocAndInitData(m_zv, m_array_length, vid);
  allocAndInitData(m_zz, m_array_length, vid);
}

void HYDRO_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_zzout, m_array_length);
  checksum[vid] += calcChecksum(m_zrout, m_array_length);
}

void HYDRO_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_zrout);
  deallocData(m_zzout);
  deallocData(m_za);
  deallocData(m_zb);
  deallocData(m_zm);
  deallocData(m_zp);
  deallocData(m_zq);
  deallocData(m_zr);
  deallocData(m_zu);
  deallocData(m_zv);
  deallocData(m_zz);
}

} // end namespace lcals
} // end namespace rajaperf
