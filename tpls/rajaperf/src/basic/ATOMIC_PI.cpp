//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


ATOMIC_PI::ATOMIC_PI(const RunParams& params)
  : KernelBase(rajaperf::Basic_ATOMIC_PI, params)
{
  setDefaultSize(3000);
  setDefaultReps(10000);

  setVariantDefined( Kokkos_Lambda );



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

ATOMIC_PI::~ATOMIC_PI()
{
}

void ATOMIC_PI::setUp(VariantID vid)
{
  m_dx = 1.0 / double(getRunSize());
  allocAndInitDataConst(m_pi, 1, 0.0, vid);
  m_pi_init = 0.0;
}

void ATOMIC_PI::updateChecksum(VariantID vid)
{
  std::cout << "Value is "<<*m_pi<<std::endl;
  checksum[vid] += Checksum_type(*m_pi);
}

void ATOMIC_PI::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_pi);
}

} // end namespace basic
} // end namespace rajaperf
