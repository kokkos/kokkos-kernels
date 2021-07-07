//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"


namespace rajaperf
{
namespace polybench
{


POLYBENCH_GEMVER::POLYBENCH_GEMVER(const RunParams& params)
  : KernelBase(rajaperf::Polybench_GEMVER, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps = 0;
  switch(lsizespec) {
    case Mini:
      m_n=40;
      run_reps = 200;
      break;
    case Small:
      m_n=120;
      run_reps = 200;
      break;
    case Medium:
      m_n=400;
      run_reps = 20;
      break;
    case Large:
      m_n=2000;
      run_reps = 20;
      break;
    case Extralarge:
      m_n=4000;
      run_reps = 5;
      break;
    default:
      m_n=800;
      run_reps = 40;
      break;
  }

  setDefaultSize(m_n*m_n + m_n*m_n + m_n + m_n*m_n);
  setDefaultReps(run_reps);

  m_alpha = 1.5;
  m_beta = 1.2;

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

POLYBENCH_GEMVER::~POLYBENCH_GEMVER()
{
}

void POLYBENCH_GEMVER::setUp(VariantID vid)
{
  (void) vid;

  allocAndInitData(m_A, m_n * m_n, vid);
  allocAndInitData(m_u1, m_n, vid);
  allocAndInitData(m_v1, m_n, vid);
  allocAndInitData(m_u2, m_n, vid);
  allocAndInitData(m_v2, m_n, vid);
  allocAndInitDataConst(m_w, m_n, 0.0, vid);
  allocAndInitData(m_x, m_n, vid);
  allocAndInitData(m_y, m_n, vid);
  allocAndInitData(m_z, m_n, vid);
}

void POLYBENCH_GEMVER::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_w, m_n);
}

void POLYBENCH_GEMVER::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_A);
  deallocData(m_u1);
  deallocData(m_v1);
  deallocData(m_u2);
  deallocData(m_v2);
  deallocData(m_w);
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace basic
} // end namespace rajaperf
