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

#include <Kokkos_Core.hpp>

#ifndef KOKKOSSPARSE_SPMV_HPP_
#define KOKKOSSPARSE_SPMV_HPP_

namespace KokkosSparse {

enum SPMVAlgorithm {
  SPMV_DEFAULT,
  SPMV_FAST_SETUP,
  SPMV_NATIVE,
  SPMV_MERGE_PATH
}

template <class AMatrix, class XVector, class YVector>
class SPMVHandle
{
  SPMVHandle(SPMVAlgorithm algo_ = SPMV_DEFAULT)
    : algo(algo_)
  {}

  SPMVAlgorithm algo;

};

}

#endif
