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

/// \file KokkosSparse_CooMatrix.hpp
/// \brief Local sparse matrix interface
///
/// This file provides KokkosSparse::CooMatrix.  This implements a
/// local (no MPI) sparse matrix stored in coordinate ("Coo") format
/// which is also known as ivj or triplet format.

#ifndef KOKKOS_SPARSE_COOMATRIX_HPP_
#define KOKKOS_SPARSE_COOMATRIX_HPP_

#include "Kokkos_Core.hpp"
#include "KokkosKernels_Error.hpp"
#include <sstream>
#include <stdexcept>

namespace KokkosSparse {
/// \class CooMatrix
///
/// \brief Coordinate format implementation of a sparse matrix.
///
/// \tparam RowView    The type of row index view.
/// \tparam ColumnView The type of column index view.
/// \tparam DataView   The type of data view.
/// \tparam Device     The Kokkos Device type.
/// \tparam MemoryTraits Traits describing how Kokkos manages and
///   accesses data.  The default parameter suffices for most users.
///
/// "Coo" stands for "coordinate format".
template <class RowView, class ColumnView, class DataView, class Device>
class CooMatrix {
 public:
  using execution_space   = typename Device::execution_space;
  using memory_space      = typename Device::memory_space;
  using data_type         = typename DataView::non_const_value_type;
  using const_data_type   = typename DataView::const_value_type;
  using row_type          = typename RowView::non_const_value_type;
  using const_row_type    = typename RowView::const_value_type;
  using column_type       = typename ColumnView::non_const_value_type;
  using const_column_type = typename ColumnView::const_value_type;
  using size_type         = size_t;

  static_assert(std::is_integral_v<row_type>,
                "RowView::value_type must be an integral.");
  static_assert(std::is_integral_v<column_type>,
                "ColumnView::value_type must be an integral.");

 private:
  size_type m_num_rows, m_num_cols;

 public:
  RowView row;
  ColumnView col;
  DataView data;

  /// \brief Default constructor; constructs an empty sparse matrix.
  KOKKOS_INLINE_FUNCTION
  CooMatrix() : m_num_rows(0), m_num_cols(0) {}

  // clang-format off
  /// \brief Constructor that accepts a column indicies view, row indices view, and
  ///        values view.
  ///
  /// The matrix will store and use the column indices, rows indices, and values
  /// directly (by view, not by deep copy).
  ///
  /// \param nrows   [in] The number of rows.
  /// \param ncols   [in] The number of columns.
  /// \param row_in  [in] The row indexes.
  /// \param col_in  [in] The column indexes.
  /// \param data_in [in] The values.
  // clang-format on
  CooMatrix(size_type nrows, size_type ncols, RowView row_in, ColumnView col_in,
            DataView data_in)
      : m_num_rows(nrows),
        m_num_cols(ncols),
        row(row_in),
        col(col_in),
        data(data_in) {
    if (data.extent(0) != row.extent(0) || row.extent(0) != col.extent(0)) {
      std::ostringstream os;
      os << "data.extent(0): " << data.extent(0) << " != "
         << "row.extent(0): " << row.extent(0) << " != "
         << "col.extent(0): " << col.extent(0) << ".";
      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  }

  //! The number of columns in the sparse matrix.
  KOKKOS_INLINE_FUNCTION size_type numCols() const { return m_num_cols; }

  //! The number of rows in the sparse matrix.
  KOKKOS_INLINE_FUNCTION size_type numRows() const { return m_num_rows; }

  //! The number of stored entries in the sparse matrix, including zeros.
  KOKKOS_INLINE_FUNCTION size_type nnz() const {
    assert(data.extent(0) == row.extent(0) == col.extent(0) &&
           "Error lengths of RowView != ColView != DataView");
    return data.extent(0);
  }
};

/// \class is_coo_matrix
/// \brief is_coo_matrix<T>::value is true if T is a CooMatrix<...>, false
/// otherwise
template <typename>
struct is_coo_matrix : public std::false_type {};
template <typename... P>
struct is_coo_matrix<CooMatrix<P...>> : public std::true_type {};
template <typename... P>
struct is_coo_matrix<const CooMatrix<P...>> : public std::true_type {};

}  // namespace KokkosSparse
#endif
