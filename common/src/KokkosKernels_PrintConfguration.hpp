//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef _KOKKOSKERNELS_PRINT_CONFIGURATION_HPP
#define _KOKKOSKERNELS_PRINT_CONFIGURATION_HPP

#include "KokkosKernels_config.h"

#include <iostream>
#include <list>

namespace KokkosKernels {
constexpr std::string_view KernelsVersionKey= "Kernels Version";
constexpr std::string_view EnabledTPLsNamesKey= "Enabled TPLs names";

namespace {
void print_enabled_tpls(std::ostream& os) {
  std::list<std::string> tpls;
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
  tpls.emplace_back("LAPACK");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
  tpls.emplace_back("BLAS");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CBLAS
  tpls.emplace_back("CBLAS");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACKE
  tpls.emplace_back("LAPACKE");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
  tpls.emplace_back("SUPERLU");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
  tpls.emplace_back("CHOLMOD");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
  tpls.emplace_back("MKL");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
  tpls.emplace_back("CUBLAS");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  tpls.emplace_back("CUSPARSE");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
  tpls.emplace_back("ROCBLAS");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCPARSE
  tpls.emplace_back("ROCPARSE");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
  tpls.emplace_back("METIS");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ARMPL
  tpls.emplace_back("ARMPL");
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
  tpls.emplace_back("MAGMA");
#endif
  if(!tpls.empty()){
    auto tplsIte = tpls.cbegin();
    os << *tplsIte;
    ++tplsIte;
    for(; tplsIte != tpls.cend(); ++tplsIte) {
      os << ";" << *tplsIte ;
    }
  }
}

void print_version(std::ostream& os) {
    os << KernelsVersionKey<< ": "<< KOKKOSKERNELS_VERSION <<'\n';
}

}  // namespace

void print_configuration(std::ostream& os) {
    print_version(os);

    os << EnabledTPLsNamesKey << ": ";
    print_enabled_tpls(os);
    os << "\n";
}

}  // namespace KokkosKernels
#endif // _KOKKOSKERNELS_PRINT_CONFIGURATION_HPP
