//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewStreamAdd.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{

void ViewStreamAdd::runSeqVariant(VariantID vid)
{
}

void ViewStreamAdd::runOpenMPVariant(VariantID vid) {}
void ViewStreamAdd::runCudaVariant(VariantID vid) {}
void ViewStreamAdd::runHipVariant(VariantID vid) {}
void ViewStreamAdd::runOpenMPTargetVariant(VariantID vid){}

} // end namespace basic
} // end namespace rajaperf
