//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ViewAllocate.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace kokkos_mechanics
{

void ViewAllocate::runSeqVariant(VariantID vid)
{
}

void ViewAllocate::runOpenMPVariant(VariantID vid) {}
void ViewAllocate::runCudaVariant(VariantID vid) {}
void ViewAllocate::runHipVariant(VariantID vid) {}
void ViewAllocate::runOpenMPTargetVariant(VariantID vid){}

} // end namespace basic
} // end namespace rajaperf
