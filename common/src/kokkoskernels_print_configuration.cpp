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

#include "kokkoskernels_print_configuration.hpp"
#include "KokkosKernels_config.h"

#include <iostream>

namespace {
void print_enabled_tpls(std::ostream& os) {
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
       os << "LAPACK" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
       os << "BLAS" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CBLAS
       os << "CBLAS" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACKE
       os << "LAPACKE" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_SUPERLU
       os << "SUPERLU" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CHOLMOD
       os << "CHOLMOD" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
       os << "MKL" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
       os << "CUBLAS" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
       os << "CUSPARSE" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
       os << "ROCBLAS" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCPARSE
       os << "ROCPARSE" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
       os << "METIS" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_ARMPL
       tpls << "ARMPL" << ";";
#endif
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
       tpls << "MAGMA" << ";";
#endif
}

void print_version(std::ostream& os) {
    os << "Kernels Version: "<< KOKKOSKERNELS_VERSION <<'\n';
}

}  // namespace

void KokkosKernels::print_configuration(std::ostream& os) {
    print_version(os);

    os << "Enabled TPLs names:\n";
    print_enabled_tpls(os);

}

