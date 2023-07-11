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

/// \file KokkosSparse_CrsMatrix_traversal.hpp
/// \brief Traversal method to access all entries in a CrsMatrix
///
/// blah

#ifndef KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP
#define KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP

#include "Kokkos_Core.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

#include "KokkosSparse_CrsMatrix_traversal_impl.hpp"

namespace KokkosSparse {
namespace Experimental {

template <class execution_space, class crsmatrix_type, class functor_type>
void crsmatrix_traversal(const execution_space& space,
                         const crsmatrix_type& matrix, functor_type& functor) {
  // Choose between device and host implementation
  if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>()) {
    KokkosSparse::Impl::crsmatrix_traversal_on_gpu(space, matrix, functor);
  } else {
    KokkosSparse::Impl::crsmatrix_traversal_on_host(space, matrix, functor);
  }
}

template <class crsmatrix_type, class functor_type>
void crsmatrix_traversal(const crsmatrix_type& matrix, functor_type& functor) {
  using execution_space = typename crsmatrix_type::execution_space;
  execution_space space{};
  crsmatrix_traversal(space, matrix, functor);
}

}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP
