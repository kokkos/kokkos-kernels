//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE


NESTED_INIT::NESTED_INIT(const RunParams& params)
  : KernelBase(rajaperf::Basic_NESTED_INIT, params)
{
  m_ni = 500;
  m_nj = 500;
  m_nk = m_nk_init = 50;

  setDefaultSize(m_ni * m_nj * m_nk);
  setDefaultReps(100);

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

  setVariantDefined( Kokkos_Lambda );


}

NESTED_INIT::~NESTED_INIT()
{
}

void NESTED_INIT::setUp(VariantID vid)
{
  m_nk = m_nk_init * static_cast<Real_type>( getRunSize() ) / getDefaultSize();
  m_array_length = m_ni * m_nj * m_nk;

  allocAndInitDataConst(m_array, m_array_length, 0.0, vid);
}

void NESTED_INIT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_array, m_array_length);
}

void NESTED_INIT::tearDown(VariantID vid)
{
  (void) vid;
  RAJA::free_aligned(m_array);
  m_array = 0;
}

} // end namespace basic
} // end namespace rajaperf
