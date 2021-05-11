//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf 
{
namespace polybench
{

 
POLYBENCH_MVT::POLYBENCH_MVT(const RunParams& params)
  : KernelBase(rajaperf::Polybench_MVT, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0; 
  switch(lsizespec) {
    case Mini:
      m_N=40;
      run_reps = 10000;
      break;
    case Small:
      m_N=120;
      run_reps = 1000;
      break;
    case Medium:
      m_N=1000;
      run_reps = 100;
      break;
    case Large:
      m_N=2000;
      run_reps = 40;
      break;
    case Extralarge:
      m_N=4000;
      run_reps = 10;
      break;
    default:
      m_N=4000;
      run_reps = 10;
      break;
  }

  setDefaultSize( 2*m_N*m_N );
  setDefaultReps(run_reps);

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

POLYBENCH_MVT::~POLYBENCH_MVT() 
{

}

void POLYBENCH_MVT::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_y1, m_N, vid);
  allocAndInitData(m_y2, m_N, vid);
  allocAndInitData(m_A, m_N * m_N, vid);
  allocAndInitDataConst(m_x1, m_N, 0.0, vid);
  allocAndInitDataConst(m_x2, m_N, 0.0, vid);
}

void POLYBENCH_MVT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x1, m_N);
  checksum[vid] += calcChecksum(m_x2, m_N);
}

void POLYBENCH_MVT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x1);
  deallocData(m_x2);
  deallocData(m_y1);
  deallocData(m_y2);
  deallocData(m_A);
}

} // end namespace polybench
} // end namespace rajaperf
