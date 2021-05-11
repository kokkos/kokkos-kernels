//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_FLOYD_WARSHALL::POLYBENCH_FLOYD_WARSHALL(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FLOYD_WARSHALL, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_N=60;
      run_reps = 100000;
      break;
    case Small:
      m_N=180;
      run_reps = 1000;
      break;
    case Medium:
      m_N=500;
      run_reps = 100;
      break;
    case Large:
      m_N=2800;
      run_reps = 1;
      break;
    case Extralarge:
      m_N=5600;
      run_reps = 1;
      break;
    default:
      m_N=300;
      run_reps = 60;
      break;
  }

  setDefaultSize( m_N*m_N*m_N );
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
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );
}

POLYBENCH_FLOYD_WARSHALL::~POLYBENCH_FLOYD_WARSHALL()
{

}

void POLYBENCH_FLOYD_WARSHALL::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitDataRandSign(m_pin, m_N*m_N, vid);
  allocAndInitDataConst(m_pout, m_N*m_N, 0.0, vid);
}

void POLYBENCH_FLOYD_WARSHALL::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_pout, m_N*m_N);
}

void POLYBENCH_FLOYD_WARSHALL::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_pin);
  deallocData(m_pout);
}

} // end namespace polybench
} // end namespace rajaperf
