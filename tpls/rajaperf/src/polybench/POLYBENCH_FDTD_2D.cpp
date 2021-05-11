//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"
#include "common/DataUtils.hpp"

#include <iostream>
#include <cstring>

namespace rajaperf
{
namespace polybench
{


POLYBENCH_FDTD_2D::POLYBENCH_FDTD_2D(const RunParams& params)
  : KernelBase(rajaperf::Polybench_FDTD_2D, params)
{
  SizeSpec lsizespec = KernelBase::getSizeSpec();
  int run_reps;
  switch(lsizespec) {
    case Mini:
      m_nx=20; m_ny=30; m_tsteps=20;
      run_reps = 10000;
      break;
    case Small:
      m_nx=60; m_ny=80; m_tsteps=40;
      run_reps = 500;
      break;
    case Medium:
      m_nx=200; m_ny=240; m_tsteps=100;
      run_reps = 200;
      break;
    case Large:
      m_nx=800; m_ny=1000; m_tsteps=500;
      run_reps = 1;
      break;
    case Extralarge:
      m_nx=2000; m_ny=2600; m_tsteps=1000;
      run_reps = 1;
      break;
    default:
      m_nx=800; m_ny=1000; m_tsteps=60;
      run_reps = 10;
      break;
  }

  setDefaultSize( m_tsteps * (m_ny + 3 * m_nx*m_ny) );
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

POLYBENCH_FDTD_2D::~POLYBENCH_FDTD_2D()
{

}

void POLYBENCH_FDTD_2D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_hz, m_nx * m_ny, 0.0, vid);
  allocAndInitData(m_ex, m_nx * m_ny, vid);
  allocAndInitData(m_ey, m_nx * m_ny, vid);
  allocAndInitData(m_fict, m_tsteps, vid);
}

void POLYBENCH_FDTD_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_hz, m_nx * m_ny);
}

void POLYBENCH_FDTD_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_fict);
  deallocData(m_ex);
  deallocData(m_ey);
  deallocData(m_hz);
}

} // end namespace polybench
} // end namespace rajaperf
