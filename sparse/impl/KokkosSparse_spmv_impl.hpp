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

#ifndef KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_

#include "Kokkos_InnerProductSpaceTraits.hpp"

#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Error.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "KokkosKernels_LowerBound.hpp"

#include "KokkosBlas1_scal.hpp"

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv_impl_omp.hpp"

#include "KokkosGraph_LoadBalance.hpp"
#include "KokkosGraph_MergePath_impl.hpp"

#define USE_UNSAFE

namespace KokkosSparse {
namespace Impl {

template <class InputType, class DeviceType>
struct GetCoeffView {
  typedef Kokkos::View<InputType*, Kokkos::LayoutLeft, DeviceType> view_type;
  typedef Kokkos::View<typename view_type::non_const_value_type*,
                       Kokkos::LayoutLeft, DeviceType>
      non_const_view_type;
  static non_const_view_type get_view(const InputType in, const int size) {
    non_const_view_type aview("CoeffView", size);
    if (size > 0) Kokkos::deep_copy(aview, in);
    return aview;
  }
};

template <class IT, class IL, class ID, class IM, class IS, class DeviceType>
struct GetCoeffView<Kokkos::View<IT*, IL, ID, IM, IS>, DeviceType> {
  typedef Kokkos::View<IT*, IL, ID, IM, IS> view_type;
  static Kokkos::View<IT*, IL, ID, IM, IS> get_view(
      const Kokkos::View<IT*, IL, ID, IM, IS>& in, int /*size*/) {
    return in;
  }
};

// This TransposeFunctor is functional, but not necessarily performant.
template <class AMatrix, class XVector, class YVector, bool conjugate>
struct SPMV_Transpose_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::ArithTraits<value_type> ATV;
  typedef typename YVector::non_const_value_type coefficient_type;
  typedef typename YVector::non_const_value_type y_value_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  ordinal_type rows_per_team = 0;

  SPMV_Transpose_Functor(const coefficient_type& alpha_, const AMatrix& m_A_,
                         const XVector& m_x_, const YVector& m_y_)
      : alpha(alpha_), m_A(m_A_), m_x(m_x_), m_y(m_y_) {}

  KOKKOS_INLINE_FUNCTION void operator()(const ordinal_type iRow) const {
    const auto row                = m_A.rowConst(iRow);
    const ordinal_type row_length = row.length;
    for (ordinal_type iEntry = 0; iEntry < row_length; iEntry++) {
      const value_type val =
          conjugate ? ATV::conj(row.value(iEntry)) : row.value(iEntry);
      const ordinal_type ind = row.colidx(iEntry);
      Kokkos::atomic_add(&m_y(ind),
                         static_cast<y_value_type>(alpha * val * m_x(iRow)));
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    const ordinal_type teamWork = dev.league_rank() * rows_per_team;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, rows_per_team), [&](ordinal_type loop) {
          // iRow represents a row of the matrix, so its correct type is
          // ordinal_type.
          const ordinal_type iRow = teamWork + loop;
          if (iRow >= m_A.numRows()) {
            return;
          }

          const auto row                = m_A.rowConst(iRow);
          const ordinal_type row_length = row.length;
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(dev, row_length),
              [&](ordinal_type iEntry) {
                const value_type val = conjugate ? ATV::conj(row.value(iEntry))
                                                 : row.value(iEntry);
                const ordinal_type ind = row.colidx(iEntry);
                Kokkos::atomic_add(&m_y(ind), static_cast<y_value_type>(
                                                  alpha * val * m_x(iRow)));
              });
        });
  }
};

template <class AMatrix, class XVector, class YVector, int dobeta,
          bool conjugate>
struct SPMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::ArithTraits<value_type> ATV;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  const value_type beta;
  YVector m_y;

  const ordinal_type rows_per_team;

  SPMV_Functor(const value_type alpha_, const AMatrix m_A_, const XVector m_x_,
               const value_type beta_, const YVector m_y_,
               const int rows_per_team_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        rows_per_team(rows_per_team_) {
    static_assert(static_cast<int>(XVector::rank) == 1,
                  "XVector must be a rank 1 View.");
    static_assert(static_cast<int>(YVector::rank) == 1,
                  "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iRow) const {
    using y_value_type = typename YVector::non_const_value_type;
    if (iRow >= m_A.numRows()) {
      return;
    }
    const KokkosSparse::SparseRowViewConst<AMatrix> row = m_A.rowConst(iRow);
    const ordinal_type row_length = static_cast<ordinal_type>(row.length);
    y_value_type sum              = 0;

    for (ordinal_type iEntry = 0; iEntry < row_length; iEntry++) {
      const value_type val =
          conjugate ? ATV::conj(row.value(iEntry)) : row.value(iEntry);
      sum += val * m_x(row.colidx(iEntry));
    }

    sum *= alpha;

    if (dobeta == 0) {
      m_y(iRow) = sum;
    } else {
      m_y(iRow) = beta * m_y(iRow) + sum;
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& dev) const {
    using y_value_type = typename YVector::non_const_value_type;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, rows_per_team),
        [&](const ordinal_type& loop) {
          const ordinal_type iRow =
              static_cast<ordinal_type>(dev.league_rank()) * rows_per_team +
              loop;
          if (iRow >= m_A.numRows()) {
            return;
          }
          const KokkosSparse::SparseRowViewConst<AMatrix> row =
              m_A.rowConst(iRow);
          const ordinal_type row_length = static_cast<ordinal_type>(row.length);
          y_value_type sum              = 0;

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(dev, row_length),
              [&](const ordinal_type& iEntry, y_value_type& lsum) {
                const value_type val = conjugate ? ATV::conj(row.value(iEntry))
                                                 : row.value(iEntry);
                lsum += val * m_x(row.colidx(iEntry));
              },
              sum);

          Kokkos::single(Kokkos::PerThread(dev), [&]() {
            sum *= alpha;

            if (dobeta == 0) {
              m_y(iRow) = sum;
            } else {
              m_y(iRow) = beta * m_y(iRow) + sum;
            }
          });
        });
  }
};

template <class execution_space>
int64_t spmv_launch_parameters(int64_t numRows, int64_t nnz,
                               int64_t rows_per_thread, int& team_size,
                               int& vector_length) {
  int64_t rows_per_team;
  int64_t nnz_per_row = nnz / numRows;

  if (nnz_per_row < 1) nnz_per_row = 1;

  int max_vector_length = 1;
#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<execution_space, Kokkos::Cuda>::value)
    max_vector_length = 32;
#endif
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same<execution_space, Kokkos::Experimental::HIP>::value)
    max_vector_length = 64;
#endif

  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length < max_vector_length && vector_length * 6 < nnz_per_row)
      vector_length *= 2;
  }

  // Determine rows per thread
  if (rows_per_thread < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>())
      rows_per_thread = 1;
    else {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  if (team_size < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>()) {
      team_size = 256 / vector_length;
    } else {
      team_size = 1;
    }
  }

  rows_per_team = rows_per_thread * team_size;

  if (rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc         = execution_space().concurrency();
    while ((conc * nnz_per_team * 4 > nnz) && (nnz_per_team > 256))
      nnz_per_team /= 2;
    rows_per_team = (nnz_per_team + nnz_per_row - 1) / nnz_per_row;
  }

  return rows_per_team;
}

// spmv_beta_no_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
template <class AMatrix, class XVector, class YVector, int dobeta,
          bool conjugate,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_beta_no_transpose(
    const KokkosKernels::Experimental::Controls& controls,
    typename YVector::const_value_type& alpha, const AMatrix& A,
    const XVector& x, typename YVector::const_value_type& beta,
    const YVector& y) {
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::execution_space execution_space;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }
#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
    /// serial impl
    typedef typename AMatrix::non_const_value_type value_type;
    typedef typename AMatrix::non_const_size_type size_type;
    typedef Kokkos::ArithTraits<value_type> ATV;

    const size_type* KOKKOS_RESTRICT row_map_ptr    = A.graph.row_map.data();
    const ordinal_type* KOKKOS_RESTRICT col_idx_ptr = A.graph.entries.data();
    const value_type* KOKKOS_RESTRICT values_ptr    = A.values.data();

    typename YVector::non_const_value_type* KOKKOS_RESTRICT y_ptr = y.data();
    typename XVector::const_value_type* KOKKOS_RESTRICT x_ptr     = x.data();

    const typename YVector::non_const_value_type zero(0);
    const ordinal_type nrow = A.numRows();
    if (alpha == zero) {
      if (dobeta == 0) {
        /// not working with kkosDev2_CUDA110_GCC92_cpp17/
        /// memset(y_ptr, 0, sizeof(typename YVector::value_type)*nrow);
        for (int i = 0; i < nrow; ++i) y_ptr[i] = zero;
      } else if (dobeta == 1) {
        /// so nothing
      } else {
        for (int i = 0; i < nrow; ++i) y_ptr[i] *= beta;
      }
    } else {
      for (int i = 0; i < nrow; ++i) {
        const int jbeg = row_map_ptr[i];
        const int jend = row_map_ptr[i + 1];
        int j          = jbeg;

        {
          const int jdist = (jend - jbeg) / 4;
          typename YVector::non_const_value_type tmp1(0), tmp2(0), tmp3(0),
              tmp4(0);
          for (int jj = 0; jj < jdist; ++jj) {
            const value_type value1 =
                conjugate ? ATV::conj(values_ptr[j]) : values_ptr[j];
            const value_type value2 =
                conjugate ? ATV::conj(values_ptr[j + 1]) : values_ptr[j + 1];
            const value_type value3 =
                conjugate ? ATV::conj(values_ptr[j + 2]) : values_ptr[j + 2];
            const value_type value4 =
                conjugate ? ATV::conj(values_ptr[j + 3]) : values_ptr[j + 3];
            const int col_idx1                        = col_idx_ptr[j];
            const int col_idx2                        = col_idx_ptr[j + 1];
            const int col_idx3                        = col_idx_ptr[j + 2];
            const int col_idx4                        = col_idx_ptr[j + 3];
            const typename XVector::value_type x_val1 = x_ptr[col_idx1];
            const typename XVector::value_type x_val2 = x_ptr[col_idx2];
            const typename XVector::value_type x_val3 = x_ptr[col_idx3];
            const typename XVector::value_type x_val4 = x_ptr[col_idx4];
            tmp1 += value1 * x_val1;
            tmp2 += value2 * x_val2;
            tmp3 += value3 * x_val3;
            tmp4 += value4 * x_val4;
            j += 4;
          }
          for (; j < jend; ++j) {
            const value_type value =
                conjugate ? ATV::conj(values_ptr[j]) : values_ptr[j];
            const int col_idx = col_idx_ptr[j];
            tmp1 += value * x_ptr[col_idx];
          }
          if (dobeta == 0) {
            y_ptr[i] = alpha * (tmp1 + tmp2 + tmp3 + tmp4);
          } else if (dobeta == 1) {
            y_ptr[i] += alpha * (tmp1 + tmp2 + tmp3 + tmp4);
          } else {
            const auto y_val = y_ptr[i] * beta;
            y_ptr[i]         = y_val + alpha * (tmp1 + tmp2 + tmp3 + tmp4);
          }
        }
      }
    }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  if ((std::is_same<execution_space, Kokkos::OpenMP>::value) &&
      (std::is_same<typename std::remove_cv<typename AMatrix::value_type>::type,
                    double>::value) &&
      (std::is_same<typename XVector::non_const_value_type, double>::value) &&
      (std::is_same<typename YVector::non_const_value_type, double>::value) &&
      ((int)A.graph.row_block_offsets.extent(0) ==
       (int)omp_get_max_threads() + 1) &&
      (((uintptr_t)(const void*)(x.data()) % 64) == 0) &&
      (((uintptr_t)(const void*)(y.data()) % 64) == 0) && !conjugate) {
    // Note BMK: this case is typically not called in practice even for OpenMP,
    // since it requires row_block_offsets to have been computed in the graph.
    spmv_raw_openmp_no_transpose<AMatrix, XVector, YVector>(alpha, A, x, beta,
                                                            y);
    return;
  }
#endif

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  SPMV_Functor<AMatrix, XVector, YVector, dobeta, conjugate> func(alpha, A, x,
                                                                  beta, y, 1);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule)
    Kokkos::parallel_for(
        "KokkosSparse::spmv<NoTranspose,Dynamic>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(
            0, A.numRows()),
        func);
  else
    Kokkos::parallel_for(
        "KokkosSparse::spmv<NoTranspose,Static>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
            0, A.numRows()),
        func);
}

// spmv_beta_no_transpose: version for GPU execution spaces (TeamPolicy used)
template <class AMatrix, class XVector, class YVector, int dobeta,
          bool conjugate,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_beta_no_transpose(
    const KokkosKernels::Experimental::Controls& controls,
    typename YVector::const_value_type& alpha, const AMatrix& A,
    const XVector& x, typename YVector::const_value_type& beta,
    const YVector& y) {
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::execution_space execution_space;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  int team_size           = -1;
  int vector_length       = -1;
  int64_t rows_per_thread = -1;

  // Note on 03/24/20, lbv: We can use the controls
  // here to allow the user to pass in some tunning
  // parameters.
  if (controls.isParameter("team size")) {
    team_size = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    rows_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  int64_t rows_per_team = spmv_launch_parameters<execution_space>(
      A.numRows(), A.nnz(), rows_per_thread, team_size, vector_length);
  int64_t worksets = (y.extent(0) + rows_per_team - 1) / rows_per_team;

  SPMV_Functor<AMatrix, XVector, YVector, dobeta, conjugate> func(
      alpha, A, x, beta, y, rows_per_team);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>
        policy(1, 1);
    if (team_size < 0)
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Dynamic>", policy,
                         func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>
        policy(1, 1);
    if (team_size < 0)
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, Kokkos::AUTO, vector_length);
    else
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::spmv<NoTranspose,Static>", policy,
                         func);
  }
}

// spmv_beta_transpose: version for CPU execution spaces (RangePolicy or trivial
// serial impl used)
template <class AMatrix, class XVector, class YVector, int dobeta,
          bool conjugate,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_beta_transpose(typename YVector::const_value_type& alpha,
                                const AMatrix& A, const XVector& x,
                                typename YVector::const_value_type& beta,
                                const YVector& y) {
  using ordinal_type    = typename AMatrix::non_const_ordinal_type;
  using size_type       = typename AMatrix::non_const_size_type;
  using execution_space = typename AMatrix::execution_space;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal(y, beta, y);
  }

#if defined(KOKKOS_ENABLE_SERIAL) || defined(KOKKOS_ENABLE_OPENMP) || \
    defined(KOKKOS_ENABLE_THREADS)
  {
    if (execution_space().concurrency() == 1) {
      /// serial impl
      typedef typename AMatrix::non_const_value_type value_type;
      typedef Kokkos::ArithTraits<value_type> ATV;
      const size_type* KOKKOS_RESTRICT row_map_ptr    = A.graph.row_map.data();
      const ordinal_type* KOKKOS_RESTRICT col_idx_ptr = A.graph.entries.data();
      const value_type* KOKKOS_RESTRICT values_ptr    = A.values.data();

      typename YVector::value_type* KOKKOS_RESTRICT y_ptr = y.data();
      typename XVector::value_type* KOKKOS_RESTRICT x_ptr = x.data();

      const typename YVector::non_const_value_type zero(0);
      const ordinal_type nrow = A.numRows();
      if (alpha == zero) {
        /// do nothing
      } else {
        for (int i = 0; i < nrow; ++i) {
          const int jbeg  = row_map_ptr[i];
          const int jend  = row_map_ptr[i + 1];
          const int jdist = (jend - jbeg) / 4;

          const typename XVector::const_value_type x_val = alpha * x_ptr[i];
          int j                                          = jbeg;
          for (int jj = 0; jj < jdist; ++jj) {
            const value_type value1 =
                conjugate ? ATV::conj(values_ptr[j]) : values_ptr[j];
            const value_type value2 =
                conjugate ? ATV::conj(values_ptr[j + 1]) : values_ptr[j + 1];
            const value_type value3 =
                conjugate ? ATV::conj(values_ptr[j + 2]) : values_ptr[j + 2];
            const value_type value4 =
                conjugate ? ATV::conj(values_ptr[j + 3]) : values_ptr[j + 3];
            const int col_idx1 = col_idx_ptr[j];
            const int col_idx2 = col_idx_ptr[j + 1];
            const int col_idx3 = col_idx_ptr[j + 2];
            const int col_idx4 = col_idx_ptr[j + 3];
            y_ptr[col_idx1] += value1 * x_val;
            y_ptr[col_idx2] += value2 * x_val;
            y_ptr[col_idx3] += value3 * x_val;
            y_ptr[col_idx4] += value4 * x_val;
            j += 4;
          }
          for (; j < jend; ++j) {
            const value_type value =
                conjugate ? ATV::conj(values_ptr[j]) : values_ptr[j];
            const int col_idx = col_idx_ptr[j];
            y_ptr[col_idx] += value * x_val;
          }
        }
      }
      return;
    }
  }
#endif

  typedef SPMV_Transpose_Functor<AMatrix, XVector, YVector, conjugate> OpType;
  typename AMatrix::const_ordinal_type nrow = A.numRows();
  Kokkos::parallel_for("KokkosSparse::spmv<Transpose>",
                       Kokkos::RangePolicy<execution_space>(0, nrow),
                       OpType(alpha, A, x, y));
}

// spmv_beta_transpose: version for GPU execution spaces (TeamPolicy used)
template <class AMatrix, class XVector, class YVector, int dobeta,
          bool conjugate,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_beta_transpose(typename YVector::const_value_type& alpha,
                                const AMatrix& A, const XVector& x,
                                typename YVector::const_value_type& beta,
                                const YVector& y) {
  using ordinal_type    = typename AMatrix::non_const_ordinal_type;
  using size_type       = typename AMatrix::non_const_size_type;
  using execution_space = typename AMatrix::execution_space;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal(y, beta, y);
  }

  // Assuming that no row contains duplicate entries, NNZPerRow
  // cannot be more than the number of columns of the matrix.  Thus,
  // the appropriate type is ordinal_type.
  const ordinal_type NNZPerRow = A.nnz() / A.numRows();

  int vector_length     = 1;
  int max_vector_length = 1;
#ifdef KOKKOS_ENABLE_CUDA
  if (std::is_same<execution_space, Kokkos::Cuda>::value)
    max_vector_length = 32;
#endif
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same<execution_space, Kokkos::Experimental::HIP>::value)
    max_vector_length = 64;
#endif
  while ((vector_length * 2 * 3 <= NNZPerRow) &&
         (vector_length < max_vector_length))
    vector_length *= 2;

  typedef SPMV_Transpose_Functor<AMatrix, XVector, YVector, conjugate> OpType;

  typename AMatrix::const_ordinal_type nrow = A.numRows();

  OpType op(alpha, A, x, y);

  const ordinal_type rows_per_thread =
      RowsPerThread<execution_space>(NNZPerRow);
  const ordinal_type team_size =
      Kokkos::TeamPolicy<execution_space>(rows_per_thread, Kokkos::AUTO,
                                          vector_length)
          .team_size_recommended(op, Kokkos::ParallelForTag());
  const ordinal_type rows_per_team = rows_per_thread * team_size;
  op.rows_per_team                 = rows_per_team;
  const size_type nteams           = (nrow + rows_per_team - 1) / rows_per_team;
  Kokkos::parallel_for(
      "KokkosSparse::spmv<Transpose>",
      Kokkos::TeamPolicy<execution_space>(nteams, team_size, vector_length),
      op);
}

template <class AMatrix, class XVector, class YVector, int dobeta>
static void spmv_beta(const KokkosKernels::Experimental::Controls& controls,
                      const char mode[],
                      typename YVector::const_value_type& alpha,
                      const AMatrix& A, const XVector& x,
                      typename YVector::const_value_type& beta,
                      const YVector& y) {
  if (mode[0] == NoTranspose[0]) {
    spmv_beta_no_transpose<AMatrix, XVector, YVector, dobeta, false>(
        controls, alpha, A, x, beta, y);
  } else if (mode[0] == Conjugate[0]) {
    spmv_beta_no_transpose<AMatrix, XVector, YVector, dobeta, true>(
        controls, alpha, A, x, beta, y);
  } else if (mode[0] == Transpose[0]) {
    spmv_beta_transpose<AMatrix, XVector, YVector, dobeta, false>(alpha, A, x,
                                                                  beta, y);
  } else if (mode[0] == ConjugateTranspose[0]) {
    spmv_beta_transpose<AMatrix, XVector, YVector, dobeta, true>(alpha, A, x,
                                                                 beta, y);
  } else {
    KokkosKernels::Impl::throw_runtime_exception(
        "Invalid Transpose Mode for KokkosSparse::spmv()");
  }
}

// Functor for implementing transpose and conjugate transpose sparse
// matrix-vector multiply with multivector (2-D View) input and
// output.  This functor works, but is not necessarily performant.
template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate>
struct SPMV_MV_Transpose_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type A_value_type;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename YVector::non_const_value_type coefficient_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  const coefficient_type beta;
  YVector m_y;

  const ordinal_type n;
  ordinal_type rows_per_team = 0;

  SPMV_MV_Transpose_Functor(const coefficient_type& alpha_, const AMatrix& m_A_,
                            const XVector& m_x_, const coefficient_type& beta_,
                            const YVector& m_y_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        n(m_x_.extent(1)) {}

  KOKKOS_INLINE_FUNCTION void operator()(const ordinal_type iRow) const {
    const auto row                = m_A.rowConst(iRow);
    const ordinal_type row_length = row.length;

    for (ordinal_type iEntry = 0; iEntry < row_length; iEntry++) {
      const A_value_type val =
          conjugate ? Kokkos::ArithTraits<A_value_type>::conj(row.value(iEntry))
                    : row.value(iEntry);
      const ordinal_type ind = row.colidx(iEntry);

      if (doalpha != 1) {
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
        for (ordinal_type k = 0; k < n; ++k) {
          Kokkos::atomic_add(&m_y(ind, k), static_cast<y_value_type>(
                                               alpha * val * m_x(iRow, k)));
        }
      } else {
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
        for (ordinal_type k = 0; k < n; ++k) {
          Kokkos::atomic_add(&m_y(ind, k),
                             static_cast<y_value_type>(val * m_x(iRow, k)));
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    const ordinal_type teamWork = dev.league_rank() * rows_per_team;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, rows_per_team), [&](ordinal_type loop) {
          // iRow represents a row of the matrix, so its correct type is
          // ordinal_type.
          const ordinal_type iRow = teamWork + loop;
          if (iRow >= m_A.numRows()) {
            return;
          }

          const auto row                = m_A.rowConst(iRow);
          const ordinal_type row_length = row.length;

          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(dev, row_length),
              [&](ordinal_type iEntry) {
                const A_value_type val =
                    conjugate ? Kokkos::ArithTraits<A_value_type>::conj(
                                    row.value(iEntry))
                              : row.value(iEntry);
                const ordinal_type ind = row.colidx(iEntry);

                if (doalpha != 1) {
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                  for (ordinal_type k = 0; k < n; ++k) {
                    Kokkos::atomic_add(
                        &m_y(ind, k),
                        static_cast<y_value_type>(alpha * val * m_x(iRow, k)));
                  }
                } else {
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
                  for (ordinal_type k = 0; k < n; ++k) {
                    Kokkos::atomic_add(&m_y(ind, k), static_cast<y_value_type>(
                                                         val * m_x(iRow, k)));
                  }
                }
              });
        });
  }
};

template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate>
struct SPMV_MV_LayoutLeft_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type A_value_type;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename YVector::non_const_value_type coefficient_type;

  const coefficient_type alpha;
  AMatrix m_A;
  XVector m_x;
  const coefficient_type beta;
  YVector m_y;
  //! The number of columns in the input and output MultiVectors.
  ordinal_type n;
  ordinal_type rows_per_thread;
  int vector_length;

  SPMV_MV_LayoutLeft_Functor(const coefficient_type& alpha_,
                             const AMatrix& m_A_, const XVector& m_x_,
                             const coefficient_type& beta_, const YVector& m_y_,
                             const ordinal_type rows_per_thread_,
                             int vector_length_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        beta(beta_),
        m_y(m_y_),
        n(m_x_.extent(1)),
        rows_per_thread(rows_per_thread_),
        vector_length(vector_length_) {}

  template <int UNROLL>
  KOKKOS_INLINE_FUNCTION void strip_mine(const team_member& dev,
                                         const ordinal_type& iRow,
                                         const ordinal_type& kk) const {
    y_value_type sum[UNROLL];

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
    for (int k = 0; k < UNROLL; ++k) {
      sum[k] = Kokkos::ArithTraits<y_value_type>::zero();
    }

    const auto row = m_A.rowConst(iRow);

    // The correct type of iEntry is ordinal_type, the type of the
    // number of columns in the (local) matrix.  This is because we
    // assume either that rows have no duplicate entries, or that rows
    // never have enough duplicate entries to overflow ordinal_type.

    Kokkos::parallel_for(
        Kokkos::ThreadVectorRange(dev, row.length), [&](ordinal_type iEntry) {
          const A_value_type val =
              conjugate
                  ? Kokkos::ArithTraits<A_value_type>::conj(row.value(iEntry))
                  : row.value(iEntry);
          const ordinal_type ind = row.colidx(iEntry);
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
          for (int k = 0; k < UNROLL; ++k) {
            sum[k] += val * m_x(ind, kk + k);
          }
        });

    if (doalpha == -1) {
      for (int ii = 0; ii < UNROLL; ++ii) {
        y_value_type sumt;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(dev, vector_length),
            [&](ordinal_type, y_value_type& lsum) {
              // in this context, sum[ii] is a partial sum ii on one of the
              // vector lanes.
              lsum -= sum[ii];
            },
            sumt);
        sum[ii] = sumt;
        // that was an all-reduce, so sum[ii] is the same on every vector lane
      }
    } else {
      for (int ii = 0; ii < UNROLL; ++ii) {
        y_value_type sumt;
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange(dev, vector_length),
            [&](ordinal_type, y_value_type& lsum) {
              // in this context, sum[ii] is a partial sum ii on one of the
              // vector lanes.
              lsum += sum[ii];
            },
            sumt);
        if (doalpha == 1)
          sum[ii] = sumt;
        else
          sum[ii] = sumt * alpha;
      }
    }

    if (dobeta == 0) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, UNROLL),
                           [&](ordinal_type k) { m_y(iRow, kk + k) = sum[k]; });
    } else if (dobeta == 1) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, UNROLL),
                           [&](ordinal_type k) {
                             m_y(iRow, kk + k) = m_y(iRow, kk + k) + sum[k];
                           });
    } else if (dobeta == -1) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(dev, UNROLL),
                           [&](ordinal_type k) {
                             m_y(iRow, kk + k) = -m_y(iRow, kk + k) + sum[k];
                           });
    } else {
      Kokkos::parallel_for(
          Kokkos::ThreadVectorRange(dev, UNROLL), [&](ordinal_type k) {
            m_y(iRow, kk + k) = beta * m_y(iRow, kk + k) + sum[k];
          });
    }
  }

  template <int UNROLL>
  KOKKOS_INLINE_FUNCTION void strip_mine(const ordinal_type& iRow,
                                         const ordinal_type& kk) const {
    y_value_type sum[UNROLL];

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
    for (int k = 0; k < UNROLL; ++k) {
      sum[k] = Kokkos::ArithTraits<y_value_type>::zero();
    }

    const auto row = m_A.rowConst(iRow);

    // The correct type of iEntry is ordinal_type, the type of the
    // number of columns in the (local) matrix.  This is because we
    // assume either that rows have no duplicate entries, or that rows
    // never have enough duplicate entries to overflow ordinal_type.

    for (ordinal_type iEntry = 0; iEntry < row.length; iEntry++) {
      const A_value_type val =
          conjugate ? Kokkos::ArithTraits<A_value_type>::conj(row.value(iEntry))
                    : row.value(iEntry);
      const ordinal_type ind = row.colidx(iEntry);
#ifdef KOKKOS_ENABLE_PRAGMA_UNROLL
#pragma unroll
#endif
      for (int k = 0; k < UNROLL; ++k) {
        if (doalpha == 1)
          sum[k] += val * m_x(ind, kk + k);
        else if (doalpha == -1)
          sum[k] -= val * m_x(ind, kk + k);
        else
          sum[k] += alpha * val * m_x(ind, kk + k);
      }
    }

    if (dobeta == 0) {
      for (ordinal_type k = 0; k < UNROLL; k++) m_y(iRow, kk + k) = sum[k];
    } else if (dobeta == 1) {
      for (ordinal_type k = 0; k < UNROLL; k++)
        m_y(iRow, kk + k) = m_y(iRow, kk + k) + sum[k];
    } else if (dobeta == -1) {
      for (ordinal_type k = 0; k < UNROLL; k++)
        m_y(iRow, kk + k) = -m_y(iRow, kk + k) + sum[k];
    } else {
      for (ordinal_type k = 0; k < UNROLL; k++)
        m_y(iRow, kk + k) = beta * m_y(iRow, kk + k) + sum[k];
    }
  }

  KOKKOS_INLINE_FUNCTION void strip_mine_1(const team_member& dev,
                                           const ordinal_type& iRow) const {
    const auto row = m_A.rowConst(iRow);

    // The correct type of iEntry is ordinal_type, the type of the
    // number of columns in the (local) matrix.  This is because we
    // assume either that rows have no duplicate entries, or that rows
    // never have enough duplicate entries to overflow ordinal_type.

    y_value_type sum;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(dev, row.length),
        [&](ordinal_type iEntry, y_value_type& lsum) {
          const A_value_type val =
              conjugate
                  ? Kokkos::ArithTraits<A_value_type>::conj(row.value(iEntry))
                  : row.value(iEntry);
          lsum += val * m_x(row.colidx(iEntry), 0);
        },
        sum);
    Kokkos::single(Kokkos::PerThread(dev), [&]() {
      if (doalpha == -1) {
        sum = -sum;
      } else if (doalpha * doalpha != 1) {
        sum *= alpha;
      }

      if (dobeta == 0) {
        m_y(iRow, 0) = sum;
      } else if (dobeta == 1) {
        m_y(iRow, 0) += sum;
      } else if (dobeta == -1) {
        m_y(iRow, 0) = -m_y(iRow, 0) + sum;
      } else {
        m_y(iRow, 0) = beta * m_y(iRow, 0) + sum;
      }
    });
  }

  KOKKOS_INLINE_FUNCTION void strip_mine_1(const ordinal_type& iRow) const {
    const auto row = m_A.rowConst(iRow);

    // The correct type of iEntry is ordinal_type, the type of the
    // number of columns in the (local) matrix.  This is because we
    // assume either that rows have no duplicate entries, or that rows
    // never have enough duplicate entries to overflow ordinal_type.

    y_value_type sum = y_value_type();
    for (ordinal_type iEntry = 0; iEntry < row.length; iEntry++) {
      const A_value_type val =
          conjugate ? Kokkos::ArithTraits<A_value_type>::conj(row.value(iEntry))
                    : row.value(iEntry);
      sum += val * m_x(row.colidx(iEntry), 0);
    }
    if (doalpha == -1) {
      sum = -sum;
    } else if (doalpha != 1) {
      sum *= alpha;
    }

    if (dobeta == 0) {
      m_y(iRow, 0) = sum;
    } else if (dobeta == 1) {
      m_y(iRow, 0) += sum;
    } else if (dobeta == -1) {
      m_y(iRow, 0) = -m_y(iRow, 0) + sum;
    } else {
      m_y(iRow, 0) = beta * m_y(iRow, 0) + sum;
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const ordinal_type& iRow) const {
    // mfh 20 Mar 2015, 07 Jun 2016: This is ordinal_type because it
    // needs to have the same type as n.
    ordinal_type kk = 0;

#ifdef KOKKOS_FAST_COMPILE
    for (; kk + 4 <= n; kk += 4) {
      strip_mine<4>(dev, iRow, kk);
    }
    for (; kk < n; ++kk) {
      strip_mine<1>(dev, iRow, kk);
    }
#else
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if ((n > 8) && (n % 8 == 1)) {
      strip_mine<9>(iRow, kk);
      kk += 9;
    }
    for (; kk + 8 <= n; kk += 8) strip_mine<8>(iRow, kk);
    if (kk < n) {
      switch (n - kk) {
#else   // NOT a GPU
    if ((n > 16) && (n % 16 == 1)) {
      strip_mine<17>(iRow, kk);
      kk += 17;
    }

    for (; kk + 16 <= n; kk += 16) {
      strip_mine<16>(iRow, kk);
    }

    if (kk < n) {
      switch (n - kk) {
        case 15: strip_mine<15>(iRow, kk); break;

        case 14: strip_mine<14>(iRow, kk); break;

        case 13: strip_mine<13>(iRow, kk); break;

        case 12: strip_mine<12>(iRow, kk); break;

        case 11: strip_mine<11>(iRow, kk); break;

        case 10: strip_mine<10>(iRow, kk); break;

        case 9: strip_mine<9>(iRow, kk); break;

        case 8: strip_mine<8>(iRow, kk); break;
#endif  // if/else: __CUDA_ARCH__ or __HIP_DEVICE_COMPILE__
        case 7: strip_mine<7>(iRow, kk); break;

        case 6: strip_mine<6>(iRow, kk); break;

        case 5: strip_mine<5>(iRow, kk); break;

        case 4: strip_mine<4>(iRow, kk); break;

        case 3: strip_mine<3>(iRow, kk); break;

        case 2: strip_mine<2>(iRow, kk); break;

        case 1: strip_mine_1(iRow); break;
      }
    }
#endif  // KOKKOS_FAST_COMPILE
  }

  KOKKOS_INLINE_FUNCTION void operator()(const team_member& dev) const {
    for (ordinal_type loop = 0; loop < rows_per_thread; ++loop) {
      // iRow indexes over (local) rows of the matrix, so its correct
      // type is ordinal_type.

      const ordinal_type iRow =
          (dev.league_rank() * dev.team_size() + dev.team_rank()) *
              rows_per_thread +
          loop;
      if (iRow >= m_A.numRows()) {
        return;
      }

      // mfh 20 Mar 2015, 07 Jun 2016: This is ordinal_type because it
      // needs to have the same type as n.
      ordinal_type kk = 0;

#ifdef KOKKOS_FAST_COMPILE
      for (; kk + 4 <= n; kk += 4) {
        strip_mine<4>(dev, iRow, kk);
      }
      for (; kk < n; ++kk) {
        strip_mine<1>(dev, iRow, kk);
      }
#else
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      if ((n > 8) && (n % 8 == 1)) {
        strip_mine<9>(dev, iRow, kk);
        kk += 9;
      }
      for (; kk + 8 <= n; kk += 8) strip_mine<8>(dev, iRow, kk);
      if (kk < n) {
        switch (n - kk) {
#else   // NOT a GPU
      if ((n > 16) && (n % 16 == 1)) {
        strip_mine<17>(dev, iRow, kk);
        kk += 17;
      }

      for (; kk + 16 <= n; kk += 16) {
        strip_mine<16>(dev, iRow, kk);
      }

      if (kk < n) {
        switch (n - kk) {
          case 15: strip_mine<15>(dev, iRow, kk); break;

          case 14: strip_mine<14>(dev, iRow, kk); break;

          case 13: strip_mine<13>(dev, iRow, kk); break;

          case 12: strip_mine<12>(dev, iRow, kk); break;

          case 11: strip_mine<11>(dev, iRow, kk); break;

          case 10: strip_mine<10>(dev, iRow, kk); break;

          case 9: strip_mine<9>(dev, iRow, kk); break;

          case 8: strip_mine<8>(dev, iRow, kk); break;
#endif  // if/else: __CUDA_ARCH__ or __HIP_DEVICE_COMPILE__
          case 7: strip_mine<7>(dev, iRow, kk); break;

          case 6: strip_mine<6>(dev, iRow, kk); break;

          case 5: strip_mine<5>(dev, iRow, kk); break;

          case 4: strip_mine<4>(dev, iRow, kk); break;

          case 3: strip_mine<3>(dev, iRow, kk); break;

          case 2: strip_mine<2>(dev, iRow, kk); break;

          case 1: strip_mine_1(dev, iRow); break;
        }
      }
#endif  // KOKKOS_FAST_COMPILE
    }
  }
};

// spmv_alpha_beta_mv_no_transpose: version for CPU execution spaces
// (RangePolicy)
template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_alpha_beta_mv_no_transpose(
    const typename YVector::non_const_value_type& alpha, const AMatrix& A,
    const XVector& x, const typename YVector::non_const_value_type& beta,
    const YVector& y) {
  using ordinal_type = typename AMatrix::non_const_ordinal_type;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }
  if (doalpha == 0) {
    if (dobeta != 1) {
      KokkosBlas::scal(y, beta, y);
    }
    return;
  } else {
    // Assuming that no row contains duplicate entries, NNZPerRow
    // cannot be more than the number of columns of the matrix.  Thus,
    // the appropriate type is ordinal_type.
    const ordinal_type NNZPerRow = A.nnz() / A.numRows();
    const int vector_length      = 1;

#ifndef KOKKOS_FAST_COMPILE  // This uses templated functions on doalpha and
                             // dobeta and will produce 16 kernels

    typedef SPMV_MV_LayoutLeft_Functor<AMatrix, XVector, YVector, doalpha,
                                       dobeta, conjugate>
        OpType;
    OpType op(alpha, A, x, beta, y,
              RowsPerThread<typename AMatrix::execution_space>(NNZPerRow),
              vector_length);

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    Kokkos::parallel_for(
        "KokkosSparse::spmv<MV,NoTranspose>",
        Kokkos::RangePolicy<typename AMatrix::execution_space>(0, nrow), op);

#else   // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for
        // alpha/beta

    typedef SPMV_MV_LayoutLeft_Functor<AMatrix, XVector, YVector, 2, 2,
                                       conjugate>
        OpType;

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    OpType op(alpha, A, x, beta, y,
              RowsPerThread<typename AMatrix::execution_space>(NNZPerRow),
              vector_length);

    Kokkos::parallel_for(
        "KokkosSparse::spmv<MV,NoTranspose>",
        Kokkos::RangePolicy<typename AMatrix::execution_space>(0, nrow), op);
#endif  // KOKKOS_FAST_COMPILE
  }
}

// spmv_alpha_beta_mv_no_transpose: version for GPU execution spaces
// (TeamPolicy)
template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_alpha_beta_mv_no_transpose(
    const typename YVector::non_const_value_type& alpha, const AMatrix& A,
    const XVector& x, const typename YVector::non_const_value_type& beta,
    const YVector& y) {
  using ordinal_type = typename AMatrix::non_const_ordinal_type;
  using size_type    = typename AMatrix::non_const_size_type;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }
  if (doalpha == 0) {
    if (dobeta != 1) {
      KokkosBlas::scal(y, beta, y);
    }
    return;
  } else {
    // Assuming that no row contains duplicate entries, NNZPerRow
    // cannot be more than the number of columns of the matrix.  Thus,
    // the appropriate type is ordinal_type.
    const ordinal_type NNZPerRow = A.nnz() / A.numRows();

    ordinal_type vector_length = 1;
    while ((vector_length * 2 * 3 <= NNZPerRow) && (vector_length < 8))
      vector_length *= 2;

#ifndef KOKKOS_FAST_COMPILE  // This uses templated functions on doalpha and
                             // dobeta and will produce 16 kernels

    typedef SPMV_MV_LayoutLeft_Functor<AMatrix, XVector, YVector, doalpha,
                                       dobeta, conjugate>
        OpType;
    OpType op(alpha, A, x, beta, y,
              RowsPerThread<typename AMatrix::execution_space>(NNZPerRow),
              vector_length);

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    const ordinal_type rows_per_thread =
        RowsPerThread<typename AMatrix::execution_space>(NNZPerRow);
    const ordinal_type team_size =
        Kokkos::TeamPolicy<typename AMatrix::execution_space>(
            rows_per_thread, Kokkos::AUTO, vector_length)
            .team_size_recommended(op, Kokkos::ParallelForTag());
    const ordinal_type rows_per_team = rows_per_thread * team_size;
    const size_type nteams = (nrow + rows_per_team - 1) / rows_per_team;
    Kokkos::parallel_for("KokkosSparse::spmv<MV,NoTranspose>",
                         Kokkos::TeamPolicy<typename AMatrix::execution_space>(
                             nteams, team_size, vector_length),
                         op);

#else   // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for
        // alpha/beta

    typedef SPMV_MV_LayoutLeft_Functor<AMatrix, XVector, YVector, 2, 2,
                                       conjugate>
        OpType;

    typename AMatrix::const_ordinal_type nrow = A.numRows();

    OpType op(alpha, A, x, beta, y,
              RowsPerThread<typename AMatrix::execution_space>(NNZPerRow),
              vector_length);

    const ordinal_type rows_per_thread =
        RowsPerThread<typename AMatrix::execution_space>(NNZPerRow);
    const ordinal_type team_size =
        Kokkos::TeamPolicy<typename AMatrix::execution_space>(
            rows_per_thread, Kokkos::AUTO, vector_length)
            .team_size_recommended(op, Kokkos::ParallelForTag());
    const ordinal_type rows_per_team = rows_per_thread * team_size;
    const size_type nteams = (nrow + rows_per_team - 1) / rows_per_team;
    Kokkos::parallel_for("KokkosSparse::spmv<MV,NoTranspose>",
                         Kokkos::TeamPolicy<typename AMatrix::execution_space>(
                             nteams, team_size, vector_length),
                         op);
#endif  // KOKKOS_FAST_COMPILE
  }
}

// spmv_alpha_beta_mv_transpose: version for CPU execution spaces (RangePolicy)
template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_alpha_beta_mv_transpose(
    const typename YVector::non_const_value_type& alpha, const AMatrix& A,
    const XVector& x, const typename YVector::non_const_value_type& beta,
    const YVector& y) {
  using ordinal_type = typename AMatrix::non_const_ordinal_type;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal(y, beta, y);
  }

  if (doalpha != 0) {
#ifndef KOKKOS_FAST_COMPILE  // This uses templated functions on doalpha and
                             // dobeta and will produce 16 kernels

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector, doalpha,
                                      dobeta, conjugate>
        OpType;
    OpType op(alpha, A, x, beta, y);

    const ordinal_type nrow = A.numRows();
    Kokkos::parallel_for(
        "KokkosSparse::spmv<MV,Transpose>",
        Kokkos::RangePolicy<typename AMatrix::execution_space>(0, nrow), op);

#else  // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for
       // alpha/beta

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector, 2, 2,
                                      conjugate, SizeType>
        OpType;

    typename AMatrix::const_ordinal_type nrow = A.numRows();
    Kokkos::parallel_for(
        "KokkosSparse::spmv<MV,Transpose>",
        Kokkos::RangePolicy<typename AMatrix::execution_space>(0, nrow), op);

#endif  // KOKKOS_FAST_COMPILE
  }
}

// spmv_alpha_beta_mv_transpose: version for GPU execution spaces (TeamPolicy)
template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta,
          bool conjugate,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename AMatrix::execution_space>()>::type* = nullptr>
static void spmv_alpha_beta_mv_transpose(
    const typename YVector::non_const_value_type& alpha, const AMatrix& A,
    const XVector& x, const typename YVector::non_const_value_type& beta,
    const YVector& y) {
  using ordinal_type = typename AMatrix::non_const_ordinal_type;
  using size_type    = typename AMatrix::non_const_size_type;

  if (A.numRows() <= static_cast<ordinal_type>(0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  if (dobeta != 1) {
    KokkosBlas::scal(y, beta, y);
  }

  if (doalpha != 0) {
    // Assuming that no row contains duplicate entries, NNZPerRow
    // cannot be more than the number of columns of the matrix.  Thus,
    // the appropriate type is ordinal_type.
    const ordinal_type NNZPerRow =
        static_cast<ordinal_type>(A.nnz() / A.numRows());

    ordinal_type vector_length = 1;
    // Transpose functor uses atomics which can't be vectorized on CPU
    while ((vector_length * 2 * 3 <= NNZPerRow) && (vector_length < 8))
      vector_length *= 2;

#ifndef KOKKOS_FAST_COMPILE  // This uses templated functions on doalpha and
                             // dobeta and will produce 16 kernels

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector, doalpha,
                                      dobeta, conjugate>
        OpType;
    OpType op(alpha, A, x, beta, y);

    const ordinal_type nrow = A.numRows();
    const ordinal_type rows_per_thread =
        RowsPerThread<typename AMatrix::execution_space>(NNZPerRow);
    const ordinal_type team_size =
        Kokkos::TeamPolicy<typename AMatrix::execution_space>(
            rows_per_thread, Kokkos::AUTO, vector_length)
            .team_size_recommended(op, Kokkos::ParallelForTag());
    const ordinal_type rows_per_team = rows_per_thread * team_size;
    op.rows_per_team                 = rows_per_team;
    const size_type nteams = (nrow + rows_per_team - 1) / rows_per_team;
    Kokkos::parallel_for("KokkosSparse::spmv<MV,Transpose>",
                         Kokkos::TeamPolicy<typename AMatrix::execution_space>(
                             nteams, team_size, vector_length),
                         op);

#else  // KOKKOS_FAST_COMPILE this will only instantiate one Kernel for
       // alpha/beta

    typedef SPMV_MV_Transpose_Functor<AMatrix, XVector, YVector, 2, 2,
                                      conjugate, SizeType>
        OpType;

    typename AMatrix::const_ordinal_type nrow = A.numRows();
    OpType op(alpha, A, x, beta, y);

    const ordinal_type rows_per_thread =
        RowsPerThread<typename AMatrix::execution_space>(NNZPerRow);
    const ordinal_type team_size =
        Kokkos::TeamPolicy<typename AMatrix::execution_space>(
            rows_per_thread, Kokkos::AUTO, vector_length)
            .team_size_recommended(op, Kokkos::ParallelForTag());
    const ordinal_type rows_per_team = rows_per_thread * team_size;
    op.rows_per_team                 = rows_per_team;
    const size_type nteams = (nrow + rows_per_team - 1) / rows_per_team;
    Kokkos::parallel_for("KokkosSparse::spmv<MV,Transpose>",
                         Kokkos::TeamPolicy<typename AMatrix::execution_space>(
                             nteams, team_size, vector_length),
                         op);

#endif  // KOKKOS_FAST_COMPILE
  }
}

template <class AMatrix, class XVector, class YVector, int doalpha, int dobeta>
static void spmv_alpha_beta_mv(
    const char mode[], const typename YVector::non_const_value_type& alpha,
    const AMatrix& A, const XVector& x,
    const typename YVector::non_const_value_type& beta, const YVector& y) {
  if (mode[0] == NoTranspose[0]) {
    spmv_alpha_beta_mv_no_transpose<AMatrix, XVector, YVector, doalpha, dobeta,
                                    false>(alpha, A, x, beta, y);
  } else if (mode[0] == Conjugate[0]) {
    spmv_alpha_beta_mv_no_transpose<AMatrix, XVector, YVector, doalpha, dobeta,
                                    true>(alpha, A, x, beta, y);
  } else if (mode[0] == Transpose[0]) {
    spmv_alpha_beta_mv_transpose<AMatrix, XVector, YVector, doalpha, dobeta,
                                 false>(alpha, A, x, beta, y);
  } else if (mode[0] == ConjugateTranspose[0]) {
    spmv_alpha_beta_mv_transpose<AMatrix, XVector, YVector, doalpha, dobeta,
                                 true>(alpha, A, x, beta, y);
  } else {
    KokkosKernels::Impl::throw_runtime_exception(
        "Invalid Transpose Mode for KokkosSparse::spmv()");
  }
}

template <class AMatrix, class XVector, class YVector, int doalpha>
void spmv_alpha_mv(const char mode[],
                   const typename YVector::non_const_value_type& alpha,
                   const AMatrix& A, const XVector& x,
                   const typename YVector::non_const_value_type& beta,
                   const YVector& y) {
  typedef typename YVector::non_const_value_type coefficient_type;
  typedef Kokkos::ArithTraits<coefficient_type> KAT;

  if (beta == KAT::zero()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 0>(mode, alpha, A, x,
                                                              beta, y);
  } else if (beta == KAT::one()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 1>(mode, alpha, A, x,
                                                              beta, y);
  } else if (beta == -KAT::one()) {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, -1>(mode, alpha, A,
                                                               x, beta, y);
  } else {
    spmv_alpha_beta_mv<AMatrix, XVector, YVector, doalpha, 2>(mode, alpha, A, x,
                                                              beta, y);
  }
}

/* Merge-based SpMV

   One thread is assigned to each non-zero.
   Threads atomically add into the product vector
*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMerge {
  typedef KokkosGraph::LoadBalanceResult<typename AMatrix::row_map_type> LBR;
  typedef typename LBR::assignment_type assignment_type;
  typedef typename LBR::rank_type rank_type;
  typedef typename YVector::non_const_value_type y_value_type;

  struct Functor {
    Functor(assignment_type& _assignment, rank_type& _rank,
            const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
            const YVector& _y)
        : assignment(_assignment),
          rank(_rank),
          alpha(_alpha),
          A(_A),
          x(_x),
          y(_y) {}

    assignment_type assignment;
    rank_type rank;
    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;

    KOKKOS_INLINE_FUNCTION void operator()(size_t i) const {
      if (i >= A.nnz()) {
        return;
      }

      typename AMatrix::non_const_ordinal_type row  = assignment(i);
      typename AMatrix::non_const_size_type nzInRow = rank(i);
      typename AMatrix::non_const_size_type ji = A.graph.row_map(row) + nzInRow;
      typename AMatrix::non_const_ordinal_type col = A.graph.entries(ji);

      Kokkos::atomic_add(
          &y(row), static_cast<y_value_type>(alpha * A.values(ji) * x(col)));
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    LBR lbr = KokkosGraph::load_balance_exclusive(A.graph.row_map, A.nnz());
    KokkosBlas::scal(y, beta, y);
    Functor functor(lbr.assignment, lbr.rank, alpha, A, x, y);
    Kokkos::parallel_for("SpmvMerge::spmv", A.nnz(), functor);
  }
};

/* Merge-based SpMV
   Each thread is assigned to a chunk of non-zeros
   Whenever a thread reaches a row-boundary, atomically increment the
   corresponding Y entry
*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMerge2 {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;

  struct Functor {
    Functor(const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
            const YVector& _y)
        : alpha(_alpha), A(_A), x(_x), y(_y) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;

    KOKKOS_INLINE_FUNCTION void operator()(size_t i) const {
      typedef typename AMatrix::size_type size_type;
      typedef typename AMatrix::ordinal_type ordinal_type;

      // iota(i) -> i
      KokkosKernels::Impl::Iota<size_t, size_t> iota(A.nnz());

      const size_type pathLength = A.numRows() + A.nnz();
      const size_type chunkSize = (pathLength + exec_space::concurrency() - 1) /
                                  exec_space::concurrency();
      const size_type d = i * chunkSize;  // diagonal

      // remove leading 0 from row_map
      Kokkos::View<typename AMatrix::row_map_type::value_type*,
                   typename AMatrix::row_map_type::device_type::memory_space,
                   Kokkos::MemoryUnmanaged>
          rowMapTail(&A.graph.row_map(1), A.graph.row_map.size() - 1);

      // beginning of my chunk
      auto dsr = KokkosGraph::Impl::diagonal_search(rowMapTail, iota, d);
      const ordinal_type myRowStart = dsr.ai;
      const size_type myNnzStart    = dsr.bi;
      // end of my chunk
      dsr = KokkosGraph::Impl::diagonal_search(rowMapTail, iota, d + chunkSize);
      const ordinal_type myRowEnd = dsr.ai;
      const size_type myNnzEnd    = dsr.bi;

      // if (myRowEnd >= A.numRows()) {
      //   printf("TOO MANY ROWS d=%lu, %d,%lu -> %d,%lu\n", d, myRowStart,
      //   myNnzStart, myRowEnd, myNnzEnd);
      // }
      // if (myNnzEnd >= A.nnz()) {
      //   printf("TOO MANY NNZ d=%lu, %d,%lu -> %d,%lu\n", d, myRowStart,
      //   myNnzStart, myRowEnd, myNnzEnd);
      // }

      // if (true) {
      //   printf("d=%lu, %d,%lu -> %d,%lu\n", d, myRowStart, myNnzStart,
      //   myRowEnd, myNnzEnd);
      // }

      // first row the first nnz is in
      ordinal_type row = myRowStart;
      size_type ni     = myNnzStart;

      // do the rest of the first row (might start in the middle of it)
      y_value_type acc = 0;
      for (; ni < A.graph.row_map(row + 1) && ni < myNnzEnd; ++ni) {
        typename AMatrix::non_const_ordinal_type col = A.graph.entries(ni);
        acc += alpha * A.values(ni) * x(col);
      }
      Kokkos::atomic_add(
          &y(row),
          acc);  // other threads may contribute to this partial product

      // do remaining rows.
      for (++row; row <= myRowEnd && row < A.numRows(); ++row) {
        acc = 0;
        for (ni = A.graph.row_map(row);
             ni < myNnzEnd && ni < A.graph.row_map(row + 1); ++ni) {
          typename AMatrix::non_const_ordinal_type col = A.graph.entries(ni);
          acc += alpha * A.values(ni) * x(col);
        }
        // if this is our last row, other threads may contribute as well
        // otherwise, we're doing the whole row, so no need for atomics
        if (row < myRowEnd) {
          y(row) += acc;
        } else {
          Kokkos::atomic_add(&y(row), acc);
        }
      }
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    const int numThreads = exec_space::concurrency();

    KokkosBlas::scal(y, beta, y);
    Functor functor(alpha, A, x, y);
    Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Static>> policy(0, numThreads);
    Kokkos::parallel_for("SpmvMerge2::spmv", policy, functor);
  }
};

/* Merge-based SpMV

  Each team is assigned to a chunk of non-zeros
  Team uses diagonal search to find it's chunk
  Team collaborates on partial products

*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMerge3 {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename XVector::non_const_value_type x_value_type;
  typedef typename AMatrix::non_const_value_type A_value_type;

  typedef Kokkos::TeamPolicy<exec_space> policy_type;
  typedef typename policy_type::member_type team_member;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::value_type*,
      typename AMatrix::row_map_type::device_type::memory_space,
      Kokkos::MemoryUnmanaged>
      um_row_map_type;

  typedef typename AMatrix::size_type size_type;
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef KokkosKernels::Impl::Iota<size_type, size_type> iota_type;

  typedef KokkosGraph::Impl::DiagonalSearchResult<
      typename um_row_map_type::size_type, typename iota_type::size_type>
      DSR;

  // results of a lower-bound and upper-bound diagonal search
  struct Chunk {
    DSR lb;  // lower bound
    DSR ub;  // upper bound
  };

  struct Functor {
    Functor(const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
            const YVector& _y)
        : alpha(_alpha), A(_A), x(_x), y(_y) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;

    KOKKOS_INLINE_FUNCTION void operator()(const team_member& thread) const {
      um_row_map_type rowEnds(&A.graph.row_map(1), A.graph.row_map.size() - 1);

      // single thread determines the beginning and end of the team chunk
      Chunk chunk;
      Kokkos::single(
          Kokkos::PerTeam(thread),
          [&](Chunk& myChunk) {
            // iota(i) -> i
            iota_type iota(A.nnz());

            const size_type pathLength = A.numRows() + A.nnz();
            const size_type chunkSize =
                (pathLength + thread.league_size() - 1) / thread.league_size();
            const size_type d = thread.league_rank() * chunkSize;  // diagonal
            const size_type dEnd =
                KOKKOSKERNELS_MACRO_MIN(d + chunkSize, pathLength);
            myChunk.lb = KokkosGraph::Impl::diagonal_search(rowEnds, iota, d);
            myChunk.ub =
                KokkosGraph::Impl::diagonal_search(rowEnds, iota, dEnd);
          },
          chunk);
      const ordinal_type teamRowStart = chunk.lb.ai;
      const size_type teamNnzStart    = chunk.lb.bi;
      const ordinal_type teamRowEnd   = chunk.ub.ai;
      const size_type teamNnzEnd      = chunk.ub.bi;

      for (ordinal_type row = teamRowStart;
           row <= teamRowEnd && row < A.numRows(); ++row) {
        y_value_type acc = 0;

        // where the non-zeros start/end in this row
        const size_type nb =
            KOKKOSKERNELS_MACRO_MAX(teamNnzStart, A.graph.row_map(row));
        const size_type ne = KOKKOSKERNELS_MACRO_MIN(teamNnzEnd, rowEnds(row));

#if 1
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(thread, nb, ne),
            [&](size_type& ni, y_value_type& lacc) {
              typename AMatrix::non_const_ordinal_type col =
                  A.graph.entries(ni);
              lacc += A.values(ni) * x(col);
            },
            acc);
#else
        // split the non-zeros in this row into three regions
        // nb....nx.....ny....ne
        //       ^ first aligned address
        //             ^ last aligned address
        // nb <= nx <= ne
        // nx <= ny <= ne
        // assume the entries and values arrays have a reasonable alignment
        // themselves
        constexpr unsigned ALIGN_TARGET = 64;
        static_assert(sizeof(ordinal_type) <= ALIGN_TARGET, "");
        static_assert(sizeof(A_value_type) <= ALIGN_TARGET, "");
        constexpr unsigned COLS_PER_ALIGN = ALIGN_TARGET / sizeof(ordinal_type);
        constexpr unsigned VALS_PER_ALIGN = ALIGN_TARGET / sizeof(A_value_type);
        constexpr unsigned PER_ALIGN =
            COLS_PER_ALIGN > VALS_PER_ALIGN ? COLS_PER_ALIGN : VALS_PER_ALIGN;
        size_type nx =
            ((nb + PER_ALIGN - 1) / PER_ALIGN) * PER_ALIGN;  // nb <= nx
        nx           = KOKKOSKERNELS_MACRO_MIN(nx, ne);      // nb <= nx <= ne
        size_type ny = (ne / PER_ALIGN) * PER_ALIGN;         // ny <= ne
        ny           = KOKKOSKERNELS_MACRO_MAX(ny, nx);      // nx <= ny <= ne

        const A_value_type* __restrict__ valuesNx =
            KokkosKernels::Impl::assume_aligned(&A.values(nx), ALIGN_TARGET);
        const ordinal_type* __restrict__ colsNx =
            KokkosKernels::Impl::assume_aligned(&A.graph.entries(nx),
                                                ALIGN_TARGET);

        // head
        y_value_type hacc;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(thread, nb, nx),
            [&](size_type& ni, y_value_type& lacc) {
              typename AMatrix::non_const_ordinal_type col =
                  A.graph.entries(ni);
              lacc += A.values(ni) * x(col);
            },
            hacc);
        acc += hacc;

        // body (aligned)
        y_value_type bacc;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(thread, 0, ny - nx),
            [&](size_type& i, y_value_type& lacc) {
              typename AMatrix::non_const_ordinal_type col = colsNx[i];
              lacc += valuesNx[i] * x(col);
            },
            bacc);
        acc += bacc;

        // tail
        y_value_type tacc;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(thread, ny, ne),
            [&](size_type& ni, y_value_type& lacc) {
              typename AMatrix::non_const_ordinal_type col =
                  A.graph.entries(ni);
              lacc += A.values(ni) * x(col);
            },
            tacc);
        acc += tacc;
#endif
        // on the margins other teams may contribute to a row as well
        Kokkos::single(Kokkos::PerTeam(thread), [&]() {
          if (row > teamRowStart && row < teamRowEnd) {
            y(row) += alpha * acc;
          } else {
            Kokkos::atomic_add(&y(row), alpha * acc);
          }
        });
      }
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    int teamSize;
    int vectorSize;
    if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>()) {
      teamSize   = 32;
      vectorSize = 1;
    } else {
      teamSize   = 1;
      vectorSize = 8;
    }

    KokkosBlas::scal(y, beta, y);
    int leagueSize = (exec_space::concurrency() + teamSize - 1) / teamSize;
    policy_type policy(exec_space(), leagueSize, teamSize, vectorSize);
    Functor functor(alpha, A, x, y);
    Kokkos::parallel_for("SpmvMerge3::spmv", policy, functor);
  }
};

/* Merge-based SpMV

  Each team is assigned to a chunk of non-zeros
  Team uses diagonal search to find it's chunk
  Team collaborates across non-zeros, uses atomics to accumulate partial
  products

*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMerge4 {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename XVector::non_const_value_type x_value_type;
  typedef typename AMatrix::non_const_value_type A_value_type;

  typedef Kokkos::TeamPolicy<exec_space> policy_type;
  typedef typename policy_type::member_type team_member;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::value_type*,
      typename AMatrix::row_map_type::device_type::memory_space,
      Kokkos::MemoryUnmanaged>
      um_row_map_type;

  typedef typename AMatrix::size_type size_type;
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef KokkosKernels::Impl::Iota<size_type, size_type> iota_type;

  typedef KokkosGraph::Impl::DiagonalSearchResult<
      typename um_row_map_type::size_type, typename iota_type::size_type>
      DSR;

  // results of a lower-bound and upper-bound diagonal search
  struct Chunk {
    DSR lb;  // lower bound
    DSR ub;  // upper bound
  };

  struct Functor {
    Functor(
        const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
        const YVector& _y,
        const ordinal_type rowsAtOnce /*products resident in scratch per team*/
        )
        : alpha(_alpha), A(_A), x(_x), y(_y), rowsAtOnce_(rowsAtOnce) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;
    ordinal_type rowsAtOnce_;

    KOKKOS_INLINE_FUNCTION void operator()(const team_member& thread) const {
      // single thread determines the beginning and end of the team chunk
      Chunk chunk;
      Kokkos::single(
          Kokkos::PerTeam(thread),
          [&](Chunk& myChunk) {
            // iota(i) -> i
            iota_type iota(A.nnz());

            // remove leading 0 from row_map
            um_row_map_type rowMapTail(&A.graph.row_map(1),
                                       A.graph.row_map.size() - 1);

            const size_type pathLength = A.numRows() + A.nnz();
            const size_type chunkSize =
                (pathLength + thread.league_size() - 1) / thread.league_size();
            const size_type d = thread.league_rank() * chunkSize;  // diagonal
            myChunk.lb =
                KokkosGraph::Impl::diagonal_search(rowMapTail, iota, d);
            myChunk.ub = KokkosGraph::Impl::diagonal_search(rowMapTail, iota,
                                                            d + chunkSize);
          },
          chunk);
      const size_type myNnzStart =
          chunk.lb.bi;  // the first nnz this team will handle
      const size_type myNnzEnd =
          chunk.ub.bi;  // one-past the last nnz this team will handle
      const ordinal_type myRowStart =
          chunk.lb.ai;  // <= the row than the first nnz is in
      const ordinal_type myRowEnd =
          chunk.ub.ai;  // >= the row than the last nnz is in

      // unmanaged subview into rowmap
      um_row_map_type rowMapSlice(
          &A.graph.row_map(myRowStart),
          myRowEnd - myRowStart + 1 /*inclusive of myRowEnd*/);

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(thread, myNnzStart, myNnzEnd),
          [&](size_type& ni) {
            // look up the row this non-zero is in
            // find the smallest row offset > ni, meaning ni is in the previous
            // row we know ni is in [myRowStart, myRowEnd] from the diagonal
            // search lower_bound returns the index of the first element in a
            // view that does not satisfy `pred(element, value)` we want the
            // index to the first row-map entry where element > ni **is**
            // satisfied, which is the first row-map entry where element <= ni
            // is not satisfied, so our predicate is `<=`

            ordinal_type row =
                KokkosKernels::lower_bound_thread(
                    rowMapSlice, ni, KokkosKernels::LTE<ordinal_type>()) +
                myRowStart;
            // row is the first row that is too large,
            // so we are in the previous row
            --row;

            typename AMatrix::non_const_ordinal_type col = A.graph.entries(ni);
            const A_value_type val                       = A.values(ni);

            Kokkos::atomic_add(&y(row), alpha * val * x(col));
          });
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    KokkosBlas::scal(y, beta, y);

    int teamSize;
    int vectorSize;
    if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>()) {
      teamSize   = 128;
      vectorSize = 1;
    } else {
      teamSize   = 1;
      vectorSize = 4;
    }

    const size_type pathLength = A.numRows() + A.nnz();

    int leagueSize = (exec_space::concurrency() + teamSize - 1) / teamSize;
    if (leagueSize > pathLength) leagueSize = pathLength;

    const size_type chunkSize = (pathLength + leagueSize - 1) / leagueSize;

    {
      static bool printOnce = false;
      if (!printOnce) {
        std::cerr << "pathLength=" << pathLength << " chunkSize=" << chunkSize
                  << " leagueSize=" << leagueSize << std::endl;
        printOnce = true;
      }
    }

    const size_t rowsAtOnce = 1024;

    policy_type policy(exec_space(), leagueSize, teamSize, vectorSize);
    Functor functor(alpha, A, x, y, rowsAtOnce);
    Kokkos::parallel_for("SpmvMerge4::spmv", policy, functor);
  }
};

/* Merge-based SpMV

   Each team uses diagonal search to find the chunk


*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMerge5 {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename XVector::non_const_value_type x_value_type;
  typedef typename AMatrix::non_const_value_type A_value_type;
  typedef typename AMatrix::row_map_type::non_const_value_type
      row_map_non_const_value_type;

  typedef Kokkos::TeamPolicy<exec_space> policy_type;
  typedef typename policy_type::member_type team_member;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::value_type*,
      typename AMatrix::row_map_type::device_type::memory_space,
      Kokkos::MemoryUnmanaged>
      um_row_map_type;

  typedef typename AMatrix::size_type size_type;
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef KokkosKernels::Impl::Iota<size_type, size_type> iota_type;

  typedef KokkosGraph::Impl::DiagonalSearchResult<
      typename um_row_map_type::size_type, typename iota_type::size_type>
      DSR;

  // results of a lower-bound and upper-bound diagonal search
  struct Chunk {
    DSR lb;  // lower bound
    DSR ub;  // upper bound
  };

  struct Functor {
    Functor(
        const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
        const YVector& _y,
        const ordinal_type rowsAtOnce /*products resident in scratch per team*/
        )
        : alpha(_alpha), A(_A), x(_x), y(_y), rowsAtOnce_(rowsAtOnce) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;
    ordinal_type rowsAtOnce_;

    KOKKOS_INLINE_FUNCTION void operator()(const team_member& thread) const {
      // single thread determines the beginning and end of the team chunk
      Chunk chunk;
      Kokkos::single(
          Kokkos::PerTeam(thread),
          [&](Chunk& myChunk) {
            // iota(i) -> i
            iota_type iota(A.nnz());

            // remove leading 0 from row_map
            um_row_map_type rowMapTail(&A.graph.row_map(1),
                                       A.graph.row_map.size() - 1);

            const size_type pathLength = A.numRows() + A.nnz();
            const size_type chunkSize =
                (pathLength + thread.league_size() - 1) / thread.league_size();
            const size_type d = thread.league_rank() * chunkSize;  // diagonal
            myChunk.lb =
                KokkosGraph::Impl::diagonal_search(rowMapTail, iota, d);
            myChunk.ub = KokkosGraph::Impl::diagonal_search(rowMapTail, iota,
                                                            d + chunkSize);
          },
          chunk);
      const size_type myNnzStart =
          chunk.lb.bi;  // the first nnz this team will handle
      const size_type myNnzEnd =
          chunk.ub.bi;  // one-past the last nnz this team will handle
      const ordinal_type myRowStart =
          chunk.lb.ai;  // <= the row than the first nnz is in
      const ordinal_type myRowEnd =
          chunk.ub.ai;  // >= the row than the last nnz is in

      y_value_type* ys = (y_value_type*)thread.team_shmem().get_shmem(
          sizeof(y_value_type) * rowsAtOnce_);
      // if (thread.team_rank() == 0) {
      //   printf("ys = %p\n", ys);
      // }

      row_map_non_const_value_type* rms =
          (row_map_non_const_value_type*)thread.team_shmem().get_shmem(
              sizeof(row_map_non_const_value_type) * rowsAtOnce_);

      // apply the team to non-zeros in rowsAtOnce_ rows at a time
      for (ordinal_type chunkStart = myRowStart; chunkStart <= myRowEnd;
           chunkStart += rowsAtOnce_) {
        // chunkEnd may contain the last nnz of the row, or be past it
        // make sure chunkEnd is not past the end of the matrix
        ordinal_type chunkEnd =
            KOKKOSKERNELS_MACRO_MIN(chunkStart + rowsAtOnce_, myRowEnd);
        chunkEnd = KOKKOSKERNELS_MACRO_MIN(chunkEnd, A.numRows() - 1);

        // initialize team scratch
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, 0, rowsAtOnce_),
            [&](const ordinal_type& i) {
              // zero shared accumulator
              ys[i] = 0;

              // copy row-map into scratch
              if (i <= chunkEnd + 1 && i + chunkStart <= A.numRows()) {
                rms[i] = A.graph.row_map(chunkStart + i);
              }
            });
        thread.team_barrier();

        // figure out where in the non-zeros this group of rows starts and stops
#if 0  // global memory
        size_type chunkNnzStart = KOKKOSKERNELS_MACRO_MAX(myNnzStart, A.graph.row_map(chunkStart));
        const size_type chunkNnzEnd = KOKKOSKERNELS_MACRO_MIN(A.graph.row_map(chunkEnd+1), myNnzEnd);
#else
        size_type chunkNnzStart = KOKKOSKERNELS_MACRO_MAX(myNnzStart, rms[0]);
        const size_type chunkNnzEnd =
            KOKKOSKERNELS_MACRO_MIN(rms[chunkEnd - chunkStart + 1], myNnzEnd);
#endif

        // if (thread.team_rank() == 0) {
        //   printf("lr=%d chunkStart=%d chunkEnd=%d chunkNnzStart=%d
        //   chunkNnzEnd=%d\n",
        //     int(thread.league_rank()),
        //     int(chunkStart),
        //     int(chunkEnd),
        //     int(chunkNnzStart),
        //     int(chunkNnzEnd)
        //   );
        // }

        // unmanaged subview into rowmap
#if 0  // global memory
        um_row_map_type rowMapSlice(&A.graph.row_map(chunkStart), chunkEnd - chunkStart + 1 /*inclusive of chunkEnd*/);
#else
        um_row_map_type rowMapSlice(
            rms, chunkEnd - chunkStart + 1 /*inclusive of chunkEnd*/);
#endif
        // row partial products
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, chunkNnzStart, chunkNnzEnd),
            [&](size_type& ni) {
              // look up the row this non-zero is in
              // find the smallest row offset > ni, meaning ni is in the
              // previous row we know ni is in [chunkStart, chunkEnd]
              // lower_bound returns the index of the first element in a view
              // that does not satisfy `pred(element, value)` we want the index
              // to the first row-map entry where element > ni **is** satisfied,
              // which is the first row-map entry where element <= ni is not
              // satisfied, so our predicate is `<=`
              ordinal_type row = KokkosKernels::lower_bound_thread(
                  rowMapSlice, ni, KokkosKernels::LTE<ordinal_type>());
              // row is the first row that is too large,
              // so we are in the previous row
              --row;

              typename AMatrix::non_const_ordinal_type col =
                  A.graph.entries(ni);
              const A_value_type val = A.values(ni);
              Kokkos::atomic_add(&ys[row], val * x(col));
            });
        thread.team_barrier();

        // accumulate into global memory
        // this may be way larger than the number of rows we're working on,
        // which will write out into a crazy place
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, 0, rowsAtOnce_),
                             [&](const ordinal_type& i) {
                               ordinal_type ii = i + chunkStart;
                               if (ii < y.extent(0)) {
                                 Kokkos::atomic_add(&y(ii), alpha * ys[i]);
                               }
                             });
        thread.team_barrier();
      }
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    KokkosBlas::scal(y, beta, y);

    int teamSize;
    int vectorSize;
    if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>()) {
      teamSize   = 128;
      vectorSize = 1;
    } else {
      teamSize   = 1;
      vectorSize = 4;
    }

    const size_type pathLength = A.numRows() + A.nnz();

    // minimize the number of global memory accesses
    // int leagueSize = A.nnz() / (2 * std::log(2 * A.numRows()));
    int leagueSize = (exec_space::concurrency() + teamSize - 1) / teamSize;

    if (leagueSize > pathLength) leagueSize = pathLength;

    const size_type chunkSize = (pathLength + leagueSize - 1) / leagueSize;

    {
      static bool printOnce = false;
      if (!printOnce) {
        std::cerr << "pathLength=" << pathLength << " chunkSize=" << chunkSize
                  << " leagueSize=" << leagueSize << std::endl;
        printOnce = true;
      }
    }

    size_t rowsAtOnce = 512;
    // TODO, this causes errors but should not
    // if (rowsAtOnce > chunkSize) rowsAtOnce = chunkSize;

    policy_type policy(exec_space(), leagueSize, teamSize, vectorSize);
    policy = policy.set_scratch_size(
        0,
        Kokkos::PerTeam(sizeof(y_value_type) * rowsAtOnce  // for y accumulator
                        + sizeof(row_map_non_const_value_type) *
                              rowsAtOnce  // for row map slice
                        ));
    Functor functor(alpha, A, x, y, rowsAtOnce);
    Kokkos::parallel_for("SpmvMerge5::spmv", policy, functor);
  }
};

/*! \brief Merge-based SpMV

  Hierarchical GPU implementation
  Each team uses MergePath search to find the non-zeros and rows it is
  responsible for Each thread in the team similarly uses diagonal search within
  the team to determine which entries it will be responsible for



  The threads then atomically accumulate partial produces

*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMergeHierarchical {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename XVector::non_const_value_type x_value_type;
  typedef typename AMatrix::non_const_value_type A_value_type;
  typedef typename AMatrix::row_map_type::non_const_value_type
      row_map_non_const_value_type;

  typedef Kokkos::TeamPolicy<exec_space> policy_type;
  typedef typename policy_type::member_type team_member;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::data_type,
      typename AMatrix::row_map_type::device_type::memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>
      // Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>
      >
      um_row_map_type;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      row_map_scratch_type;

  typedef Kokkos::View<
      typename AMatrix::staticcrsgraph_type::entries_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      entries_scratch_type;

  typedef Kokkos::View<
      typename AMatrix::values_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      values_scratch_type;

  typedef Kokkos::View<
      typename YVector::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      y_scratch_type;

  typedef typename AMatrix::size_type size_type;
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef KokkosKernels::Impl::Iota<size_type, size_type> iota_type;

  typedef KokkosGraph::Impl::DiagonalSearchResult<
      typename um_row_map_type::size_type, typename iota_type::size_type>
      DSR;

  using KAT = Kokkos::Details::ArithTraits<A_value_type>;
  // using a_mag_type = typename KAT::mag_type;

  // results of a lower-bound and upper-bound diagonal search
  struct Chunk {
    DSR lb;  // lower bound
    DSR ub;  // upper bound
  };

  template <bool NONZEROS_USE_SCRATCH, bool ROWENDS_USE_SCRATCH,
            bool Y_USE_SCRATCH, bool CONJ>
  struct Functor {
    Functor(const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
            const YVector& _y, const size_type pathLengthThreadChunk)
        : alpha(_alpha),
          A(_A),
          x(_x),
          y(_y),
          pathLengthThreadChunk_(pathLengthThreadChunk) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;
    size_type pathLengthThreadChunk_;

    KOKKOS_INLINE_FUNCTION void operator()(const team_member& thread) const {
      const size_type pathLengthTeamChunk =
          thread.team_size() * pathLengthThreadChunk_;

      const size_type pathLength = A.numRows() + A.nnz();
      const size_type teamD =
          thread.league_rank() * pathLengthTeamChunk;  // diagonal
      const size_type teamDEnd =
          KOKKOSKERNELS_MACRO_MIN(teamD + pathLengthTeamChunk, pathLength);

      // iota(i) -> i
      iota_type iota(A.nnz());

      // remove leading 0 from row_map
      um_row_map_type rowEnds(&A.graph.row_map(1), A.graph.row_map.size() - 1);

#if 0
      // determine the beginning and end of the team chunk
      Chunk teamChunk;
      Kokkos::single(Kokkos::PerTeam(thread), [&](Chunk &myChunk) {
        myChunk.lb = KokkosGraph::Impl::diagonal_search(rowEnds, iota, teamD);
        myChunk.ub = KokkosGraph::Impl::diagonal_search(rowEnds, iota, teamDEnd);
      }, teamChunk);

      const size_type teamNnzBegin = teamChunk.lb.bi; // the first nnz this team will handle
      const size_type teamNnzEnd = teamChunk.ub.bi; // one-past the last nnz this team will handle
      const ordinal_type teamRowBegin = teamChunk.lb.ai; // <= the row than the first nnz is in
      const ordinal_type teamRowEnd = teamChunk.ub.ai; // >= the row than the last nnz is in
#elif 0
      DSR lb = KokkosGraph::Impl::diagonal_search(thread, rowEnds, iota, teamD);
      DSR ub =
          KokkosGraph::Impl::diagonal_search(thread, rowEnds, iota, teamDEnd);

      const size_type teamNnzBegin =
          lb.bi;  // the first nnz this team will handle
      const size_type teamNnzEnd =
          ub.bi;  // one-past the last nnz this team will handle
      const ordinal_type teamRowBegin =
          lb.ai;  // <= the row than the first nnz is in
      const ordinal_type teamRowEnd =
          ub.ai;  // >= the row than the last nnz is in

#else
      DSR lb;
      DSR ub;

      // thread 0 does the lower bound, thread 1 does the upper bound
      if (0 == thread.team_rank() || 1 == thread.team_rank()) {
        const size_type d = thread.team_rank() ? teamDEnd : teamD;
        DSR dsr = KokkosGraph::Impl::diagonal_search(rowEnds, iota, d);
        if (0 == thread.team_rank()) {
          lb = dsr;
        }
        if (1 == thread.team_rank()) {
          ub = dsr;
        }
      }
      thread.team_broadcast(lb, 0);
      thread.team_broadcast(ub, 1);
      const size_type teamNnzBegin =
          lb.bi;  // the first nnz this team will handle
      const size_type teamNnzEnd =
          ub.bi;  // one-past the last nnz this team will handle
      const ordinal_type teamRowBegin =
          lb.ai;  // <= the row than the first nnz is in
      const ordinal_type teamRowEnd =
          ub.ai;  // >= the row than the last nnz is in
#endif

      // if (teamRowBegin != teamRowEnd && thread.team_rank() == 0) {
      //   printf(
      //     "lr=%d       teamD=%d teamRow=%d-%d teamNnz=%d-%d\n",
      //     int(thread.league_rank()),
      //     int(teamD),
      //     int(teamRowBegin), int(teamRowEnd),
      //     int(teamNnzBegin), int(teamNnzEnd)
      //   );
      // }

      // team-collaborative copy of matrix data into scratch
      row_map_scratch_type rowEndsS;
      entries_scratch_type entriesS;
      values_scratch_type valuesS;
      y_scratch_type yS;

      if constexpr (ROWENDS_USE_SCRATCH) {
        rowEndsS =
            row_map_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, teamRowBegin, teamRowEnd + 1),
            [&](const ordinal_type& i) {
              rowEndsS(i - teamRowBegin) = rowEnds(i);
            });
      } else {
        (void)(rowEndsS == rowEndsS);  // set but unused, expr has no effect
      }

      if constexpr (NONZEROS_USE_SCRATCH) {
        entriesS =
            entries_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
        valuesS = values_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, teamNnzBegin, teamNnzEnd),
            [&](const ordinal_type& i) {
              valuesS(i - teamNnzBegin)  = A.values(i);
              entriesS(i - teamNnzBegin) = A.graph.entries(i);
            });
      } else {
        (void)(entriesS == entriesS);  // set but unused, expr has no effect
        (void)(valuesS == valuesS);    // set but unused, expr has no effect
      }

      if constexpr (Y_USE_SCRATCH) {
        yS = y_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, teamRowBegin, teamRowEnd + 1),
            [&](const ordinal_type& i) {
              if (i < A.numRows()) {
                yS(i - teamRowBegin) = 0;
              }
            });
      } else {
        (void)(yS == yS);  // set but unused, expr has no effect
      }

      if constexpr (ROWENDS_USE_SCRATCH || NONZEROS_USE_SCRATCH ||
                    Y_USE_SCRATCH) {
        thread.team_barrier();
      }

      // each thread determines it's location within the team chunk

      // team's view of row map is either in scratch or global
      typename std::conditional<ROWENDS_USE_SCRATCH, row_map_scratch_type,
                                um_row_map_type>::type teamRowEnds;

      if constexpr (ROWENDS_USE_SCRATCH) {
        teamRowEnds =
            row_map_scratch_type(&rowEndsS(0), teamRowEnd - teamRowBegin);
      } else {
        teamRowEnds =
            um_row_map_type(&rowEnds(teamRowBegin), teamRowEnd - teamRowBegin);
      }

      iota_type teamIota(teamNnzEnd - teamNnzBegin,
                         teamNnzBegin);  // teamNnzBegin..<teamNnzEnd

      // diagonal is local to the team's path
      size_type threadD = KOKKOSKERNELS_MACRO_MIN(
          teamD + thread.team_rank() * pathLengthThreadChunk_, pathLength);
      threadD -= teamD;

      // size_type threadDEnd = KOKKOSKERNELS_MACRO_MIN(
      //   threadD + pathLengthThreadChunk_,
      //   pathLength
      // );
      // threadDEnd -= teamD;

      Chunk threadChunk;
      threadChunk.lb =
          KokkosGraph::Impl::diagonal_search(teamRowEnds, teamIota, threadD);
      const size_type threadNnzBegin    = threadChunk.lb.bi + teamNnzBegin;
      const ordinal_type threadRowBegin = threadChunk.lb.ai + teamRowBegin;

      // threadChunk.ub = KokkosGraph::Impl::diagonal_search(teamRowEnds,
      // teamIota, threadDEnd); const size_type threadNnzEnd = threadChunk.ub.bi
      // + teamNnzBegin; const ordinal_type threadRowEnd = threadChunk.ub.ai +
      // teamRowBegin;

      // if (threadNnzBegin != threadNnzEnd) {
      //   printf(
      //     "lr=%d tr=%d teamD=%d teamRow=%d-%d teamNnz=%d-%d threadD=%d
      //     threadRow=%d-%d threadNnz=%d-%d\n", int(thread.league_rank()),
      //     int(thread.team_rank()), int(teamD), int(teamRowBegin),
      //     int(teamRowEnd), int(teamNnzBegin), int(teamNnzEnd), int(threadD),
      //     int(threadRowBegin), int(threadRowEnd),
      //     int(threadNnzBegin), int(threadNnzEnd)
      //   );
      // }

      // each thread does some accumulating
      y_value_type acc    = 0;
      ordinal_type curRow = threadRowBegin;
      size_type curNnz    = threadNnzBegin;
      for (ordinal_type i = 0;
           i < pathLengthThreadChunk_ /*some threads have less work*/ &&
           curNnz < A.nnz() + 1 && curRow < A.numRows();
           ++i) {
        size_type curRowEnd;
        if constexpr (ROWENDS_USE_SCRATCH) {
          curRowEnd = rowEndsS(curRow - teamRowBegin);
        } else {
          curRowEnd = rowEnds(curRow);
        }

        if (curNnz < curRowEnd) {
          typename AMatrix::non_const_ordinal_type col;
          A_value_type val;
          if constexpr (NONZEROS_USE_SCRATCH) {
            col = entriesS(curNnz - teamNnzBegin);
            val = valuesS(curNnz - teamNnzBegin);
          } else {
            col = A.graph.entries(curNnz);
            val = CONJ ? KAT::conj(A.values(curNnz)) : A.values(curNnz);
          }
          // printf(
          //   "lr=%d tr=%d accumulate curNnz=%d\n",
          //   int(thread.league_rank()), int(thread.team_rank()),
          //   int(curNnz)
          // );

          acc += val * x(col);
          ++curNnz;
        } else {
          // if (0 != acc) {
          //   printf(
          //     "lr=%d tr=%d store %f to curRow=%d\n",
          //     int(thread.league_rank()), int(thread.team_rank()),
          //     float(alpha * acc), int(curRow)
          //   );
          // }
          if constexpr (Y_USE_SCRATCH) {
#if defined(KOKKOS_ENABLE_HIP) && __HIP_DEVICE_COMPILE__
            unsafeAtomicAdd(&yS(curRow - teamRowBegin), alpha * acc);
#else
            Kokkos::atomic_add(&yS(curRow - teamRowBegin), alpha * acc);
#endif
          } else {
#if defined(KOKKOS_ENABLE_HIP) && __HIP_DEVICE_COMPILE__
            unsafeAtomicAdd(&y(curRow), alpha * acc);
#else
            Kokkos::atomic_add(&y(curRow), alpha * acc);
#endif
          }
          acc = 0;
          ++curRow;
        }
      }
      // save the accumulated results of a partial last row.
      // might be 0 if last row was not partial
      if (curRow < A.numRows()) {
        if constexpr (Y_USE_SCRATCH) {
          Kokkos::atomic_add(&yS(curRow - teamRowBegin), alpha * acc);
        } else {
#if defined(KOKKOS_ENABLE_HIP) && __HIP_DEVICE_COMPILE__
          unsafeAtomicAdd(&y(curRow), alpha * acc);
#else
          Kokkos::atomic_add(&y(curRow), alpha * acc);
#endif
        }
      }

      if constexpr (Y_USE_SCRATCH) {
        thread.team_barrier();

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(thread, teamRowBegin, teamRowEnd + 1),
            [&](const ordinal_type& i) {
              if (i < A.numRows()) {
                if (i > teamRowBegin && i < teamRowEnd) {
                  y(i) += yS(i - teamRowBegin);
                } else {
#if defined(KOKKOS_ENABLE_HIP) && __HIP_DEVICE_COMPILE__
                  unsafeAtomicAdd(&y(i), yS(i - teamRowBegin));
#else
                  Kokkos::atomic_add(&y(i), yS(i - teamRowBegin));
#endif
                }
              }
            });
      }
    }

    size_t team_shmem_size(int teamSize) const {
      const size_type pathLengthTeamChunk = pathLengthThreadChunk_ * teamSize;
      size_t val                          = 0;
      if constexpr (Y_USE_SCRATCH) {
        val += sizeof(y_value_type) * pathLengthTeamChunk;
      }
      if constexpr (ROWENDS_USE_SCRATCH) {
        val += sizeof(row_map_non_const_value_type) * pathLengthTeamChunk;
      }
      if constexpr (NONZEROS_USE_SCRATCH) {
        val += sizeof(ordinal_type) * pathLengthTeamChunk;
        val += sizeof(A_value_type) * pathLengthTeamChunk;
      }
      return val;
    }
  };

  /*
     collaboratively reduce N key-value pairs from each thread
     into the output views
  */
  template <typename KeyView, typename ValueView, unsigned N>
  static void reduce_by_key(const team_member& thread, ordinal_type ik[N],
                            y_value_type iv[N], const KeyView& ok,
                            const ValueView& ov) {}

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    KokkosBlas::scal(y, beta, y);

    /* determine launch parameters for different architectures
       On architectures where there is a natural execution hierarchy with true
       team scratch, we'll assign each team to use an appropriate amount of the
       scratch.

       On other architectures, just have each team do the maximal amount of work
       to amortize the cost of the diagonal search
    */
    const size_type pathLength = A.numRows() + A.nnz();
    size_type pathLengthThreadChunk;
    int teamSize;
    if constexpr (false) {
    }
#if defined(KOKKOS_ENABLE_CUDA)
    else if constexpr (std::is_same<exec_space, Kokkos::Cuda>::value) {
      pathLengthThreadChunk = 4;
      teamSize              = 128;
    }
#elif defined(KOKKOS_ENABLE_HIP)
    else if constexpr (std::is_same<exec_space, Kokkos::HIP>::value) {
      pathLengthThreadChunk = 4;
      teamSize              = 128;
    }
#else
    else if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<
                           exec_space>()) {
      pathLengthThreadChunk = 4;
      teamSize = 128;
    }
#endif
    else {
      teamSize              = 1;
      pathLengthThreadChunk = (pathLength + exec_space::concurrency() - 1) /
                              exec_space::concurrency();
    }

    const size_t pathLengthTeamChunk = pathLengthThreadChunk * teamSize;
    const int leagueSize =
        (pathLength + pathLengthTeamChunk - 1) / pathLengthTeamChunk;

    {
      static bool printOnce = false;
      if (!printOnce) {
        std::cerr << "pathLength=" << pathLength
                  << " pathLengthThreadChunk=" << pathLengthThreadChunk
                  << " pathLengthTeamChunk=" << pathLengthTeamChunk
                  << " leagueSize=" << leagueSize << std::endl;
        printOnce = true;
      }
    }

    policy_type policy(exec_space(), leagueSize, teamSize);

    /* Currently:
       Volta+ global atomics are fast, so don't use scratch for y
       CDNA2 atomics are CAS unless unsafe is explicitly requested, which only
       works on coarse-grained allocations global memory is fast for unsafe Not
       sure about Intel

       On CPU spaces, there's no real point to using scratch, just rely on the
       memory hierarchy Using scratch just increases the number of required
       atomic operations
    */
    if (KokkosSparse::NoTranspose == mode) {
      constexpr bool CONJ = false;
      using GpuOp         = Functor<true, true, false, CONJ>;
      using CpuOp         = Functor<false, false, false, CONJ>;
      using Op            = typename std::conditional<
          KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>(), GpuOp,
          CpuOp>::type;
      Op op(alpha, A, x, y, pathLengthThreadChunk);
      Kokkos::parallel_for("SpmvMergeHierarchical::spmv", policy, op);
    } else if (KokkosSparse::Conjugate == mode) {
      constexpr bool CONJ = true;
      using GpuOp         = Functor<true, true, false, CONJ>;
      using CpuOp         = Functor<false, false, false, CONJ>;
      using Op            = typename std::conditional<
          KokkosKernels::Impl::kk_is_gpu_exec_space<exec_space>(), GpuOp,
          CpuOp>::type;
      Op op(alpha, A, x, y, pathLengthThreadChunk);
      Kokkos::parallel_for("SpmvMergeHierarchical::spmv", policy, op);
    } else {
      throw std::logic_error("unsupported mode");
    }
  }
};

/* Merge-based SpMV

  Hierarchical GPU implementation
  Each team uses MergePath search to find the non-zeros and rows it is
  responsible for Each thread in the team similarly uses diagonal search within
  the team to determine which entries it will be responsible for



  The threads then atomically accumulate partial produces

*/
template <class AMatrix, class XVector, class YVector>
struct SpmvMvMergeHierarchical {
  typedef typename YVector::device_type device_type;
  typedef typename device_type::execution_space exec_space;
  typedef typename YVector::non_const_value_type y_value_type;
  typedef typename XVector::non_const_value_type x_value_type;
  typedef typename AMatrix::non_const_value_type A_value_type;
  typedef typename AMatrix::row_map_type::non_const_value_type
      row_map_non_const_value_type;

  typedef Kokkos::TeamPolicy<exec_space> policy_type;
  typedef typename policy_type::member_type team_member;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::data_type,
      typename AMatrix::row_map_type::device_type::memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>
      // Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>
      >
      um_row_map_type;

  typedef Kokkos::View<
      typename AMatrix::row_map_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      row_map_scratch_type;

  typedef Kokkos::View<
      typename AMatrix::staticcrsgraph_type::entries_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      entries_scratch_type;

  typedef Kokkos::View<
      typename AMatrix::values_type::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      values_scratch_type;

  typedef Kokkos::View<
      typename YVector::non_const_data_type,
      typename exec_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess |
                           Kokkos::Restrict>>
      y_scratch_type;

  using KAT = Kokkos::Details::ArithTraits<A_value_type>;
  // using a_mag_type = typename KAT::mag_type;

  typedef typename AMatrix::size_type size_type;
  typedef typename AMatrix::ordinal_type ordinal_type;
  typedef KokkosKernels::Impl::Iota<size_type, size_type> iota_type;

  typedef KokkosGraph::Impl::DiagonalSearchResult<
      typename um_row_map_type::size_type, typename iota_type::size_type>
      DSR;

  // results of a lower-bound and upper-bound diagonal search
  struct Chunk {
    DSR lb;  // lower bound
    DSR ub;  // upper bound
  };

  template <bool CONJ>
  struct Functor {
    Functor(const y_value_type& _alpha, const AMatrix& _A, const XVector& _x,
            const YVector& _y, const size_type pathLengthThreadChunk)
        : alpha(_alpha),
          A(_A),
          x(_x),
          y(_y),
          pathLengthThreadChunk_(pathLengthThreadChunk) {}

    y_value_type alpha;
    AMatrix A;
    XVector x;
    YVector y;
    size_type pathLengthThreadChunk_;

    template <unsigned K>
    KOKKOS_INLINE_FUNCTION void contribute_path(
        const team_member& thread, const unsigned ks,
        const row_map_scratch_type& rowEnds, ordinal_type teamRowBegin,
        const entries_scratch_type& entries, const values_scratch_type& values,
        size_type teamNnzBegin, ordinal_type threadRowBegin,
        size_type threadNnzBegin) const {
      y_value_type acc[K] = {0};  // prefer this if acc is still registers

      /* Make sure the last row is contributed out to Y
         Consider curNnz = A.nnz() - 1 (the final non-zero)
         and curRow = A.numRows() - 1 (the final row)

         curNnz < curRowEnd, so the final partial product will be accumulated,
         and then ++curNnz

         now curNnz == A.nnz() and curNnz !< curRowEnd, so the final row
         will be written out.
         Thus we need to run this loop when curNnz < A.nnz() + 1
      */
      ordinal_type curRow = threadRowBegin;
      size_type curNnz    = threadNnzBegin;
      for (ordinal_type i = 0;
           i < ordinal_type(pathLengthThreadChunk_) &&
           curNnz < A.nnz() + 1 /* write out the final row*/ &&
           curRow < A.numRows();
           ++i) {
        const size_type curRowEnd = rowEnds(curRow - teamRowBegin);

        if (curNnz < curRowEnd) {
          typename AMatrix::non_const_ordinal_type col;
          A_value_type val;
          col = entries(curNnz - teamNnzBegin);
          val = values(curNnz - teamNnzBegin);
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (unsigned k = 0; k < K; ++k) {
            acc[k] += (CONJ ? KAT::conj(val) : val) * x(col, ks + k);
          }

          ++curNnz;
        } else {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
          for (unsigned k = 0; k < K; ++k) {
            Kokkos::atomic_add(&y(curRow, ks + k), alpha * acc[k]);
            acc[k] = 0;
          }
          ++curRow;
        }
      }

      // the path may end in the middle of a row
      // save the accumulated results of a partial last row (or 0 if path ended
      // on a row boundary) may have already done the last row in the matrix in
      // which case acc will be 0 and curRow will be incremented past the
      // boundary
      if (curRow < A.numRows()) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
        for (unsigned k = 0; k < K; ++k) {
          Kokkos::atomic_add(&y(curRow, ks + k), alpha * acc[k]);
        }
      }
    }

    KOKKOS_INLINE_FUNCTION void operator()(const team_member& thread) const {
      const size_type pathLengthTeamChunk =
          thread.team_size() * pathLengthThreadChunk_;

      const size_type pathLength = A.numRows() + A.nnz();
      const size_type teamD =
          thread.league_rank() * pathLengthTeamChunk;  // diagonal
      const size_type teamDEnd =
          KOKKOSKERNELS_MACRO_MIN(teamD + pathLengthTeamChunk, pathLength);

      // iota(i) -> i
      iota_type iota(A.nnz());

      // remove leading 0 from row_map
      um_row_map_type rowEnds(&A.graph.row_map(1), A.graph.row_map.size() - 1);

      DSR lb;
      DSR ub;

      // thread 0 does the lower bound, thread 1 does the upper bound
      if (0 == thread.team_rank() || 1 == thread.team_rank()) {
        const size_type d = thread.team_rank() ? teamDEnd : teamD;
        DSR dsr = KokkosGraph::Impl::diagonal_search(rowEnds, iota, d);
        if (0 == thread.team_rank()) {
          lb = dsr;
        }
        if (1 == thread.team_rank()) {
          ub = dsr;
        }
      }
      thread.team_broadcast(lb, 0);
      thread.team_broadcast(ub, 1);
      const size_type teamNnzBegin =
          lb.bi;  // the first nnz this team will handle
      const size_type teamNnzEnd =
          ub.bi;  // one-past the last nnz this team will handle
      const ordinal_type teamRowBegin =
          lb.ai;  // <= the row than the first nnz is in
      const ordinal_type teamRowEnd =
          ub.ai;  // >= the row than the last nnz is in

      // team-collaborative copy of matrix data into scratch
      row_map_scratch_type rowEndsS;
      entries_scratch_type entriesS;
      values_scratch_type valuesS;

      rowEndsS = row_map_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(thread, teamRowBegin, teamRowEnd + 1),
          [&](const ordinal_type& i) {
            rowEndsS(i - teamRowBegin) = rowEnds(i);
          });

      entriesS = entries_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
      valuesS  = values_scratch_type(thread.team_shmem(), pathLengthTeamChunk);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(thread, teamNnzBegin, teamNnzEnd),
          [&](const ordinal_type& i) {
            valuesS(i - teamNnzBegin)  = A.values(i);
            entriesS(i - teamNnzBegin) = A.graph.entries(i);
          });
      thread.team_barrier();

      row_map_scratch_type teamRowEnds(&rowEndsS(0), teamRowEnd - teamRowBegin);

      // each thread determines it's location within the team chunk
      iota_type teamIota(teamNnzEnd - teamNnzBegin,
                         teamNnzBegin);  // teamNnzBegin..<teamNnzEnd

      // diagonal is local to the team's path
      size_type threadD = KOKKOSKERNELS_MACRO_MIN(
          teamD + thread.team_rank() * pathLengthThreadChunk_, pathLength);
      threadD -= teamD;

      // size_type threadDEnd = KOKKOSKERNELS_MACRO_MIN(
      //   threadD + pathLengthThreadChunk_,
      //   pathLength
      // );
      // threadDEnd -= teamD;

      Chunk threadChunk;
      threadChunk.lb =
          KokkosGraph::Impl::diagonal_search(teamRowEnds, teamIota, threadD);
      const size_type threadNnzBegin    = threadChunk.lb.bi + teamNnzBegin;
      const ordinal_type threadRowBegin = threadChunk.lb.ai + teamRowBegin;

      // threadChunk.ub = KokkosGraph::Impl::diagonal_search(teamRowEnds,
      // teamIota, threadDEnd); const size_type threadNnzEnd = threadChunk.ub.bi
      // + teamNnzBegin; const ordinal_type threadRowEnd = threadChunk.ub.ai +
      // teamRowBegin;

      // each thread does some accumulating
      constexpr unsigned K_CHUNK_SIZE = 4;
      unsigned ks;
      for (ks = 0; ks + K_CHUNK_SIZE < x.extent(1); ks += K_CHUNK_SIZE) {
        contribute_path<K_CHUNK_SIZE>(thread, ks, rowEndsS, teamRowBegin,
                                      entriesS, valuesS, teamNnzBegin,
                                      threadRowBegin, threadNnzBegin);
      }
      for (; ks < x.extent(1); ++ks) {
        contribute_path<1>(thread, ks, rowEndsS, teamRowBegin, entriesS,
                           valuesS, teamNnzBegin, threadRowBegin,
                           threadNnzBegin);
      }
    }

    size_t team_shmem_size(int teamSize) const {
      const size_type pathLengthTeamChunk = pathLengthThreadChunk_ * teamSize;
      size_t val                          = 0;
      val += sizeof(row_map_non_const_value_type) * pathLengthTeamChunk;
      val += sizeof(ordinal_type) * pathLengthTeamChunk;
      val += sizeof(A_value_type) * pathLengthTeamChunk;
      return val;
    }
  };

  static void spmv(const char mode[], const y_value_type& alpha,
                   const AMatrix& A, const XVector& x, const y_value_type& beta,
                   const YVector& y) {
    KokkosBlas::scal(y, beta, y);

    /* determine launch parameters for different architectures
       On architectures where there is a natural execution hierarchy with true
       team scratch, we'll assign each team to use an appropriate amount of the
       scratch.

       On other architectures, just have each team do the maximal amount of work
       to amortize the cost of the diagonal search
    */
    const size_type pathLength = A.numRows() + A.nnz();
    size_type pathLengthThreadChunk;
    int teamSize;
    if constexpr (false) {
    }
#if defined(KOKKOS_ENABLE_CUDA)
    else if constexpr (std::is_same<exec_space, Kokkos::Cuda>::value) {
      pathLengthThreadChunk = 4;
      teamSize              = 128;
    }
#elif defined(KOKKOS_ENABLE_HIP)
    else if constexpr (std::is_same<exec_space, Kokkos::HIP>::value) {
      pathLengthThreadChunk = 4;
      teamSize              = 64;
    }
#else
    else if constexpr (KokkosKernels::Impl::kk_is_gpu_exec_space<
                           exec_space>()) {
      pathLengthThreadChunk = 4;
      teamSize = 128;
    }
#endif
    else {
      teamSize              = 1;
      pathLengthThreadChunk = (pathLength + exec_space::concurrency() - 1) /
                              exec_space::concurrency();
    }

    const size_t pathLengthTeamChunk = pathLengthThreadChunk * teamSize;
    const int leagueSize =
        (pathLength + pathLengthTeamChunk - 1) / pathLengthTeamChunk;

    {
      static bool printOnce = false;
      if (!printOnce) {
        std::cerr << "SpmvMvMergeHierarchical"
                  << " pathLength=" << pathLength
                  << " pathLengthThreadChunk=" << pathLengthThreadChunk
                  << " pathLengthTeamChunk=" << pathLengthTeamChunk
                  << " leagueSize=" << leagueSize << std::endl;
        printOnce = true;
      }
    }

    policy_type policy(exec_space(), leagueSize, teamSize);

    if (KokkosSparse::NoTranspose == mode) {
      constexpr bool CONJ = false;
      using Op            = Functor<CONJ>;
      Op op(alpha, A, x, y, pathLengthThreadChunk);
      Kokkos::parallel_for("SpmvMvMergeHierarchical::spmv", policy, op);
    } else if (KokkosSparse::Conjugate == mode) {
      constexpr bool CONJ = true;
      using Op            = Functor<CONJ>;
      Op op(alpha, A, x, y, pathLengthThreadChunk);
      Kokkos::parallel_for("SpmvMvMergeHierarchical::spmv", policy, op);
    } else {
      throw std::logic_error("unsupported mode");
    }
  }
};  // SpmvMvMergeHierarchical

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_IMPL_SPMV_DEF_HPP_
