//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewStreamAdd.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf 
{
namespace kokkos_mechanics
{

// Syntax for C++ constructor
ViewStreamAdd::ViewStreamAdd(const RunParams& params)
  : KernelBase(rajaperf::KokkosMechanics_ViewStreamAdd, params)
{
  setDefaultSize(100000); 
  setDefaultReps(5000);

  setVariantDefined( Kokkos_Lambda_Seq);
  setVariantDefined( Kokkos_Lambda_OpenMP);
  setVariantDefined( Kokkos_Lambda_OpenMPTarget);
  setVariantDefined( Kokkos_Lambda_CUDA);
}
//Defining the destructor (for the struct)
ViewStreamAdd::~ViewStreamAdd() 
{
}

void ViewStreamAdd::setUp(VariantID vid)
{
  h_a = VT("host_a",getRunSize());
  h_b = VT("host_b",getRunSize());
  h_c = VT("host_c",getRunSize());
  Kokkos::deep_copy(h_a,1.0f);
  Kokkos::deep_copy(h_b,2.0f);
}

void ViewStreamAdd::updateChecksum(VariantID vid)
{
//  checksum[vid] += calcChecksum(m_y, getRunSize());
}

void ViewStreamAdd::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
