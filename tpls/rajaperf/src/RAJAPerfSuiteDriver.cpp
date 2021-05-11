//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "common/Executor.hpp"
#include "common/QuickKernelBase.hpp"
#include <iostream>

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
  // STEP 1: Create suite executor object
  //rajaperf::Executor executor(argc, argv);
  rajaperf::Executor executor(argc, argv);
  rajaperf::make_perfsuite_executor(&executor, argc, argv);
  //executor.registerKernel
  //rajaperf::RunParams params(argc, argv);
  //executor.registerGroup("Sparse");

  //executor.registerKernel("Sparse", rajaperf::make_kernel_base(
  //        "Sparse_SPMV", params, [&](const int repfact, const int size){
  //        },
  //        [&] (const int repfact, const int size) {}
  //        ));
  //  executor.registerKernel("Sparse", rajaperf::make_kernel_base(
  //          "Sparse_SPMM", params, [&](const int repfact, const int size){
  //             return std::make_tuple(1);
  //          },
  //          [&] (const int repfact, const int size, auto matrix) {
  //              // do the math using Kokkos Kernels operators
  //          }
  //  ));

  // STEP 2: Assemble kernels and variants to run
  executor.setupSuite();

  // STEP 3: Report suite run summary 
  //         (enable users to catch errors before entire suite is run)
  executor.reportRunSummary(std::cout); 

  // STEP 4: Execute suite
  executor.runSuite();

  // STEP 5: Generate suite execution reports
  executor.outputRunData();

  std::cout << "\n\nDONE!!!...." << std::endl; 

  return 0;
}
