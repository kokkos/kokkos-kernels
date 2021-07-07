//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_GEMM::POLYBENCH_GEMM(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMM, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_ni = 20; m_nj = 25; m_nk = 30;
      run_reps = 10000;
      break;
    case Small:
      m_ni = 60; m_nj = 70; m_nk = 80;
      run_reps = 1000;
      break;
    case Medium:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
    case Large:
      m_ni = 1000; m_nj = 1100; m_nk = 1200;
      run_reps = 1;
      break;
    case Extralarge:
      m_ni = 2000; m_nj = 2300; m_nk = 2600;
      run_reps = 1;
      break;
    default:
      m_ni = 200; m_nj = 220; m_nk = 240;
      run_reps = 100;
      break;
  }

  setDefaultSize( m_ni * (m_nj + m_nj*m_nk) );
  setDefaultReps(run_reps);

  m_alpha = 0.62;
  m_beta = 1.002;

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

POLYBENCH_GEMM::~POLYBENCH_GEMM()
{
}

void POLYBENCH_GEMM::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_A, m_ni * m_nk, vid);
  allocAndInitData(m_B, m_nk * m_nj, vid);
  allocAndInitDataConst(m_C, m_ni * m_nj, 0.0, vid);
}

void POLYBENCH_GEMM::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_C, m_ni * m_nj);
}

void POLYBENCH_GEMM::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_C);
}

} // end namespace polybench
} // end namespace rajaperf
