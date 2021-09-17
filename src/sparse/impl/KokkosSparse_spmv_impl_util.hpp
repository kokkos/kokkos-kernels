/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_UTIL_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_UTIL_HPP

#include "Kokkos_Core.hpp"

namespace KokkosSparse {
namespace Impl {

/// \brief Routine to verify the input arguments to avoid repeating
///
/// \param[in] mode  Flag for Normal, Conjugate, Transpose, Hermitian 
/// \param[in] A  Matrix
/// \param[in] x  Right-hand side vector (or multi-vector)
/// \param[in] y  Right-hand side vector (or multi-vector)
template <class AMatrix, class XVector, class YVector>
bool verifyArguments(const char mode[], const AMatrix &A, const XVector &x,
                     const YVector &y) {
  // Make sure that both x and y have the same rank.
  static_assert(
      static_cast<int>(XVector::rank) == static_cast<int>(YVector::rank),
      "KokkosSparse::spmv: Vector ranks do not match.");
  // Make sure that y is non-const.
  static_assert(std::is_same<typename YVector::value_type,
                             typename YVector::non_const_value_type>::value,
                "KokkosSparse::spmv: Output Vector must be non-const.");

  // Check compatibility of dimensions at run time.
  if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
      (mode[0] == KokkosSparse::Conjugate[0])) {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(x.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(y.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match: "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(y.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(x.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match (transpose): "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  }

  return true;
}

//////////////////////////////////////////////////////////

// \brief Constant setting an upper-bound for block sizes
// on the compiler-based "unrolling"
constexpr size_t bmax = 12;

// \brief "Unrolling" constant for large block-size
constexpr int unroll  = 4;

/// \brief raw_axpy  Vector operation y <- y + alpha * x
///
/// \param[in] alpha  Weight
/// \param[in] x   Input vector x (stored as an array)
/// \param[in,out] y  Output vector y  (passed as a raw pointer)
///
template <int M, typename Scalar>
inline void raw_axpy(const Scalar &alpha,
                     const std::array<Scalar, Impl::bmax> &x, Scalar *y_ptr) {
  for (int ic = 0; ic < M; ++ic) y_ptr[ic] = y_ptr[ic] + alpha * x[ic];
}

/// \brief Matrix-matrix product  Y <- Y + A * X
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param Aval  Matrix values stored in row-major fashion
/// \param lda  Leading dimension to access the next column
/// \param x_ptr  Right-hand side pointer (size Mx x k)
///               X is assumed to be stored in a column-major fashion
/// \param Ax  Left-hand side array  (size M x k)
template <int M, typename Scalar>
inline void raw_gemm_n(const Scalar *Aval, int lda,
                       const Scalar *x_ptr, int ldx, int K,
                       std::vector<Scalar> &Ax) {
  for (int ik = 0; ik < K; ++ik) {
    for (int kr = 0; kr < M; ++kr) {
      for (int ic = 0; ic < M; ++ic) {
        const auto xvalue = x_ptr[ic + ik * ldx];
        Ax[kr + ik * M] += Aval[ic + kr * lda] * xvalue;
      }
    }
  }
}

/// \brief Matrix-matrix product  Y <- Y + conj(A) * X
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param Aval  Matrix values stored in row-major fashion
/// \param lda  Leading dimension to access the next column
/// \param x_ptr  Right-hand side pointer (size Mx x k)
///               X is assumed to be stored in a column-major fashion
/// \param Ax  Left-hand side array  (size M x k)
template <int M, typename Scalar>
inline void raw_gemm_c(const Scalar *Aval, int lda,
                       const Scalar *x_ptr, int ldx, int K,
                       std::vector<Scalar> &Ax) {
  for (int ik = 0; ik < K; ++ik) {
    for (int kr = 0; kr < M; ++kr) {
      for (int ic = 0; ic < M; ++ic) {
        const auto xvalue = x_ptr[ic + ik * ldx];
        Ax[kr + ik * M] += Kokkos::ArithTraits<Scalar>::conj(Aval[ic + kr * lda]) * xvalue;
      }
    }
  }
}

/// \brief Matrix-matrix product  Y <- Y + A^T * X
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param Aval  Matrix values stored in row-major fashion
/// \param lda  Leading dimension to access the next column
/// \param x_ptr  Right-hand side pointer (size Mx x k)
///               X is assumed to be stored in a column-major fashion
/// \param Ax  Left-hand side array  (size M x k)
template <int M, typename Scalar>
inline void raw_gemm_t(const Scalar *Aval, int lda,
                       const Scalar *x_ptr, int ldx, int K,
                       std::vector<Scalar> &Ax) {
  for (int ik = 0; ik < K; ++ik) {
    for (int kr = 0; kr < M; ++kr) {
      for (int ic = 0; ic < M; ++ic) {
        const auto xvalue = x_ptr[ic + ik * ldx];
        Ax[kr + ik * M] += Aval[kr + ic * lda] * xvalue;
      }
    }
  }
}

/// \brief Matrix-matrix product  Y <- Y + A^H * X
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param Aval  Matrix values stored in row-major fashion
/// \param lda  Leading dimension to access the next column
/// \param x_ptr  Right-hand side pointer (size Mx x k)
///               X is assumed to be stored in a column-major fashion
/// \param Ax  Left-hand side array  (size M x k)
template <int M, typename Scalar>
    inline void raw_gemm_h(const Scalar *Aval, int lda,
                           const Scalar *x_ptr, int ldx, int K,
                           std::vector<Scalar> &Ax) {
  for (int ik = 0; ik < K; ++ik) {
    for (int kr = 0; kr < M; ++kr) {
      for (int ic = 0; ic < M; ++ic) {
        const auto xvalue = x_ptr[ic + ik * ldx];
        Ax[kr + ik * M] += Kokkos::ArithTraits<Scalar>::conj(Aval[kr + ic * lda]) * xvalue;
      }
    }
  }
}

/// \brief Matrix-vector product  y <- y + A * x
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param[in] Aval
/// \param[in] lda
/// \param[in] x_ptr  Right-hand side pointer (size Mx x 1)
/// \param[out] Ax  Left-hand side array  (size M x 1)
template <int M, typename Scalar, int Mx = M>
inline void raw_gemv_n(const Scalar *Aval, int lda, const Scalar *x_ptr,
                       std::array<Scalar, Impl::bmax> &Ax) {
  for (int ic = 0; ic < Mx; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (int kr = 0; kr < M; ++kr) {
      Ax[kr] += Aval[ic + kr * lda] * xvalue;
    }
  }
}

template void raw_gemv_n< Impl::bmax, std::complex<double> >(
    const std::complex<double> *Aval, int lda, const std::complex<double> *x_ptr,
    std::array<std::complex<double>, Impl::bmax> &Ax
    );

/// \brief Matrix-vector product  y <- y + conj(A) * x
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param[in] Aval
/// \param[in] lda
/// \param[in] x_ptr  Right-hand side pointer (size Mx x 1)
/// \param[out] Ax  Left-hand side array  (size M x 1)
template <int M, typename Scalar, int Mx = M>
inline void raw_gemv_c(const Scalar *Aval, int lda, const Scalar *x_ptr,
                       std::array<Scalar, Impl::bmax> &Ax) {
  for (int ic = 0; ic < Mx; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (int kr = 0; kr < M; ++kr) {
      Ax[kr] += Kokkos::ArithTraits<Scalar>::conj(Aval[ic + kr * lda]) * xvalue;
    }
  }
}

/// \brief Matrix-vector product  y <- y + A^T * x
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param[in] Aval
/// \param[in] lda
/// \param[in] x_ptr  Right-hand side pointer (size Mx x 1)
/// \param[out] Ax  Left-hand side array  (size M x 1)
template <int M, typename Scalar, int Mx = M>
inline void raw_gemv_t(const Scalar *Aval, int lda, const Scalar *x_ptr,
                       std::array<Scalar, Impl::bmax> &Ax) {
  for (int ic = 0; ic < Mx; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (int kr = 0; kr < M; ++kr) {
      Ax[kr] += Aval[kr + ic * lda] * xvalue;
    }
  }
}

/// \brief Matrix-vector product  y <- y + conj(A)^T * x
///
/// \tparam M  Number of rows in A
/// \tparam Scalar  Value type
/// \tparam Mx  Number of columns in A
/// \param[in] Aval
/// \param[in] lda
/// \param[in] x_ptr  Right-hand side pointer (size Mx x 1)
/// \param[out] Ax  Left-hand side array  (size M x 1)
template <int M, typename Scalar, int Mx = M>
inline void raw_gemv_h(const Scalar *Aval, int lda, const Scalar *x_ptr,
                       std::array<Scalar, Impl::bmax> &Ax) {
  for (int ic = 0; ic < Mx; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (int kr = 0; kr < M; ++kr) {
      Ax[kr] += Kokkos::ArithTraits<Scalar>::conj(Aval[kr + ic * lda]) * xvalue;
    }
  }
}

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_UTIL_HPP
