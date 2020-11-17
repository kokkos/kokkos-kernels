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

#ifndef _KOKKOSNEWSPGEMMIMPL_HPP
#define _KOKKOSNEWSPGEMMIMPL_HPP

#include <KokkosKernels_Utils.hpp>
#include <KokkosKernels_SimpleUtils.hpp>
#include <KokkosKernels_SparseUtils.hpp>
#include <KokkosKernels_VectorUtils.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "KokkosKernels_HashmapAccumulator.hpp"
#include "KokkosKernels_Uniform_Initialized_MemoryPool.hpp"
#include "KokkosSparse_spgemm_handle.hpp"

namespace KokkosSparse{

  namespace Impl{

    template <typename HandleType>
    class SPGEMM{
    public:

      using ExecSpace = typename HandleType::HandleExecSpace;
      using MemSpace = typename HandleType::HandleTempMemorySpace;
      using Device = Kokkos::Device<ExecSpace, MemSpace>;
      using Layout = Kokkos::LayoutLeft;  // This is a harsh assumption.
                                          // What is the best way of getting layout info 
                                          //    without templating on view types? 

      using ordinal_t = typename HandleType::nnz_lno_t;
      using offset_t = typename HandleType::size_type;
      using scalar_t = typename HandleType::nnz_scalar_t;

      using const_row_map_t = Kokkos::View<const offset_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using const_entries_t = Kokkos::View<const ordinal_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using const_values_t = Kokkos::View<const scalar_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      using row_map_t = Kokkos::View<offset_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using entries_t = Kokkos::View<ordinal_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      using values_t = Kokkos::View<scalar_t *, Layout, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    private:
      HandleType *handle;
      ordinal_t a_row_cnt;
      ordinal_t b_row_cnt;
      ordinal_t b_col_cnt;

      const_row_map_t row_mapA;
      const_entries_t entriesA;
      const_values_t valuesA;
      bool transposeA;

      const_row_map_t row_mapB;
      const_entries_t entriesB;
      const_values_t valuesB;
      bool transposeB;

      struct NumericFunctor;

      template<typename c_row_map_t>
      void numeric_impl(c_row_map_t rowmapC_,
			entries_t entriesC_,
			values_t valuesC_);

    public:
      void numeric(row_map_t &rowmapC_, entries_t &entriesC_, values_t &valuesC_) {
	numeric_impl(rowmapC_, entriesC_, valuesC_);
      };

      void numeric(const_row_map_t &rowmapC_, entries_t &entriesC_, values_t &valuesC_) {
      	numeric_impl(rowmapC_, entriesC_, valuesC_);
      };


      SPGEMM(HandleType *handle_,
	     ordinal_t m_,
	     ordinal_t n_,
	     ordinal_t k_,
	     const_row_map_t row_mapA_,
	     const_entries_t entriesA_,
	     const_values_t valuesA_,
	     bool transposeA_,
	     const_row_map_t row_mapB_,
	     const_entries_t entriesB_,
	     const_values_t valuesB_,
	     bool transposeB_):handle (handle_), a_row_cnt(m_), b_row_cnt(n_), b_col_cnt(k_),
			       row_mapA(row_mapA_), entriesA(entriesA_), valuesA(valuesA_), transposeA(transposeA_),
			       row_mapB(row_mapB_), entriesB(entriesB_), valuesB(valuesB_), transposeB(transposeB_)
      {}

    };
  }
}
#include "KokkosSparse_new_spgemm_numeric_impl.hpp"
#endif
