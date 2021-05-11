//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_JACOBI_2D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_JACOBI_2D::POLYBENCH_JACOBI_2D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_JACOBI_2D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_N=30;
      m_tsteps=20;
      run_reps = 1000;
      break;
    case Small:
      m_N=90;
      m_tsteps=40;
      run_reps = 500;
      break;
    case Medium:
      m_N=250;
      m_tsteps=100;
      run_reps = 300;
      break;
    case Large:
      m_N=1500;
      m_tsteps=20;
      run_reps = 10;
      break;
    case Extralarge:
      m_N=2800;
      m_tsteps=10;
      run_reps = 1;
      break;
    default:
      m_N=1000;
      m_tsteps=40;
      run_reps = 10;
      break;
  }

  setDefaultSize( m_tsteps * 2 * m_N * m_N );
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

POLYBENCH_JACOBI_2D::~POLYBENCH_JACOBI_2D()
{
}

void POLYBENCH_JACOBI_2D::setUp(VariantID vid)
{
  (void) vid;
  allocAndInitData(m_Ainit, m_N*m_N, vid);
  allocAndInitData(m_Binit, m_N*m_N, vid);
  allocAndInitDataConst(m_A, m_N*m_N, 0.0, vid);
  allocAndInitDataConst(m_B, m_N*m_N, 0.0, vid);
}

void POLYBENCH_JACOBI_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_A, m_N*m_N);
  checksum[vid] += calcChecksum(m_B, m_N*m_N);
}

void POLYBENCH_JACOBI_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_B);
  deallocData(m_Ainit);
  deallocData(m_Binit);
}

} // end namespace polybench
} // end namespace rajaperf
