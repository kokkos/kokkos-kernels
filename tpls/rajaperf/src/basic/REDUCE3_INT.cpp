//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <limits>

namespace rajaperf 
{
namespace basic
{


REDUCE3_INT::REDUCE3_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_REDUCE3_INT, params)
{
  setDefaultSize(1000000);
//setDefaultReps(5000);
// Set reps to low value until we resolve RAJA omp-target 
// reduction performance issues
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
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Kokkos_Lambda );



}

REDUCE3_INT::~REDUCE3_INT() 
{
}

void REDUCE3_INT::setUp(VariantID vid)
{
  allocAndInitData(m_vec, getRunSize(), vid);

  m_vsum = 0;
  m_vsum_init = 0;
  m_vmin = std::numeric_limits<Int_type>::max();
  m_vmin_init = std::numeric_limits<Int_type>::max();
  m_vmax = std::numeric_limits<Int_type>::min();
  m_vmax_init = std::numeric_limits<Int_type>::min();
}

void REDUCE3_INT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_vsum;
  checksum[vid] += m_vmin;
  checksum[vid] += m_vmax;
}

void REDUCE3_INT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_vec);
}

} // end namespace basic
} // end namespace rajaperf
