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
/// This file provides a public interface to traversal
/// methods that are used as a common and efficient way
/// to access entries in a matrix on host and/or device.

#ifndef KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP
#define KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP

#include "Kokkos_Core.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

#include "KokkosSparse_CrsMatrix_traversal_impl.hpp"

namespace KokkosSparse {
namespace Experimental {


/// \brief Public interface to sparse matrix traversal algorithm.
///
/// Loop over the entries of the input matrix and apply the functor
/// to them. The functor itself may contain its own data to save results
/// after the traversal completes.
///
/// \tparam execution_space
/// \tparam crsmatrix_type
/// \tparam functor_type
///
/// \param space [in] execution space instance that specifies where the kernel
///   will be executed.
/// \param matrix [in] the matrix to be traversed.
/// \param functor [in] a functor that is being called on each local entries
/// of the crsmatrix and that implement a user defined capabilities.
///
template <class execution_space, class crsmatrix_type, class functor_type>
void crsmatrix_traversal(const execution_space& space,
                         const crsmatrix_type& matrix, functor_type& functor) {

  // Check if a quick return can be performed
  if(!matrix.nnz()) { return; }

  // Choose between device and host implementation
  if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>()) {
    KokkosSparse::Impl::crsmatrix_traversal_on_gpu(space, matrix, functor);
  } else {
    KokkosSparse::Impl::crsmatrix_traversal_on_host(space, matrix, functor);
  }
}

/// \brief Public interface to sparse matrix traversal algorithm.
///
/// Loop over the entries of the input matrix and apply the functor
/// to them. The functor itself may contain its own data to save results
/// after the traversal completes.
///
/// \tparam crsmatrix_type
/// \tparam functor_type
///
/// \param matrix [in] the matrix to be traversed.
/// \param functor [in] a functor that is being called on each local entries
/// of the crsmatrix and that implement a user defined capabilities.
///
template <class crsmatrix_type, class functor_type>
void crsmatrix_traversal(const crsmatrix_type& matrix, functor_type& functor) {
  using execution_space = typename crsmatrix_type::execution_space;
  execution_space space{};
  crsmatrix_traversal(space, matrix, functor);
}

}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_CRSMATRIX_TRAVERSAL_HPP
