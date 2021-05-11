//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewAllocate.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace kokkos_mechanics
{

// Syntax for C++ constructor
ViewAllocate::ViewAllocate(const RunParams& params)
  : KernelBase(rajaperf::KokkosMechanics_ViewAllocate, params)
{
  setDefaultSize(100000); 
  setDefaultReps(5000);

  setVariantDefined( Kokkos_Lambda_Seq);
  setVariantDefined( Kokkos_Lambda_OpenMP);
  setVariantDefined( Kokkos_Lambda_OpenMPTarget);
  setVariantDefined( Kokkos_Lambda_CUDA);
}
//Defining the destructor (for the struct)
ViewAllocate::~ViewAllocate() 
{
}

void ViewAllocate::setUp(VariantID vid)
{
}

void ViewAllocate::updateChecksum(VariantID vid)
{
//  checksum[vid] += calcChecksum(m_y, getRunSize());
}

void ViewAllocate::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
