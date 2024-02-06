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
// Use TPL utilities for safely finalizing matrix descriptors, etc.
#include "KokkosSparse_Utils_cusparse.hpp"
#include "KokkosSparse_Utils_rocsparse.hpp"
#include "KokkosSparse_Utils_mkl.hpp"

#ifndef KOKKOSSPARSE_SPMV_HANDLE_HPP_
#define KOKKOSSPARSE_SPMV_HANDLE_HPP_

namespace KokkosSparse {

enum SPMVAlgorithm {
  SPMV_DEFAULT,
  SPMV_FAST_SETUP,
  SPMV_NATIVE,
  SPMV_MERGE_PATH
}

namespace Impl
{
  struct CuSparse9_SpMV_Data
  {
    ~CuSparse9_SpMV_Data()
    {
    }

    cusparseMatDescr_t mat;
    size_t bufferSize = 0;
    void* buffer = nullptr;
  };

  // Data used by cuSPARSE 10 and up
  struct CuSparse_SpMV_Data
  {
    ~CuSparse_SpMV_Data()
    {
    }

    cusparseSpMatDescr_t mat;
  };

  struct RocSparse_SpMV_Data
  {
    ~RocSparse_SpMV_Data()
    {
    }

    rocsparse_mat_descr mat;
    rocsparse_spmat_descr spmat;
    size_t bufferSize = 0;
    void* buffer = nullptr;
  };

  struct MKL_SpMV_Data
  {
    sparse_matrix_t mat;
    matrix_descr descr;
  };

#if defined(KOKKOS_ENABLE_SYCL) && \
  !defined(KOKKOSKERNELS_ENABLE_TPL_MKL_SYCL_OVERRIDE)
  struct OneMKL_SpMV_Data
  {
  };
#endif
}

template <class ExecutionSpace, class AMatrix>
class SPMVHandle
{
public:
  SPMVHandle(SPMVAlgorithm algo_ = SPMV_DEFAULT)
    : algo(algo_)
  {}

  ~SPMVHandle()
  {
  }

  SPMVAlgorithm get_algorithm() const {return algo};

private:
  SPMVAlgorithm algo;
};

}

#endif
