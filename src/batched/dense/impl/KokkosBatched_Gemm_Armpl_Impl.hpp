//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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
#ifndef __KOKKOSBATCHED_GEMM_ARMPL_IMPL_HPP__
#define __KOKKOSBATCHED_GEMM_ARMPL_IMPL_HPP__
#if defined(KOKKOSKERNELS_ENABLE_TPL_ARMPL)
#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
///
/// Serial Impl
/// ===========
/********************* END functor-level routines *********************/

namespace Impl {
/********************* BEGIN non-functor-level routines *********************/
// TODO: wrap this class in a macro for permutations of supported scalars.
template <class ArgTransA, class ArgTransB, class ArgBatchSzDim,
          class HandleType, class ScalarType, class AViewType, class BViewType,
          class CViewType>
class BatchedArmplGemm {
 private:
  HandleType *const __handle;
  using avt = typename AViewType::value_type;
  using bvt = typename BViewType::value_type;
  using cvt = typename CViewType::value_type;

  AViewType __A;
  avt *__Adp = nullptr;
  armpl_int_t __Ajstrd, __Aistrd, __Abstrd;

  BViewType __B;
  bvt *__Bdp = nullptr;
  armpl_int_t __Bjstrd, __Bistrd, __Bbstrd;

  CViewType __C;
  cvt *__Cdp = nullptr;
  armpl_int_t __Cjstrd, __Cistrd, __Cbstrd;

  ScalarType __alpha, __beta;
  armpl_int_t __ninter, __nbatch;

  char __transa, __transb;

  ArgTransA __transa_tag;
  ArgTransB __transb_tag;
  Trans::NoTranspose __no_trans_tag;
  ArgBatchSzDim __batch_layout_tag;

  armpl_int_t __Am, __An, __Bm, __Bn, __Cm, __Cn;

  void __unpack_views() {
    for (int ib = 0; ib < __nbatch; ++ib) {
      for (int i = 0; i < __ninter; ++i) {
        auto svA =
            subview_wrapper(__A, ib * __ninter + i, Kokkos::ALL(),
                            Kokkos::ALL(), __batch_layout_tag, __no_trans_tag);
        auto svB =
            subview_wrapper(__B, ib * __ninter + i, Kokkos::ALL(),
                            Kokkos::ALL(), __batch_layout_tag, __no_trans_tag);
        auto svC =
            subview_wrapper(__C, ib * __ninter + i, Kokkos::ALL(),
                            Kokkos::ALL(), __batch_layout_tag, __no_trans_tag);

        auto info = armpl_dge_interleave(
            __ninter, i, __Am, __An, reinterpret_cast<double *>(svA.data()),
            svA.stride(0), svA.stride(1),
            reinterpret_cast<double *>(&__Adp[__Abstrd * ib]), __Aistrd,
            __Ajstrd);
        if (info != ARMPL_STATUS_SUCCESS) {
          std::ostringstream os;
          os << "armpl_dge_interleave(A) returned:" << info << std::endl;
          Kokkos::Impl::throw_runtime_exception(os.str());
        }

        info = armpl_dge_interleave(
            __ninter, i, __Bm, __Bn, reinterpret_cast<double *>(svB.data()),
            svB.stride(0), svB.stride(1),
            reinterpret_cast<double *>(&__Bdp[__Bbstrd * ib]), __Bistrd,
            __Bjstrd);
        if (info != ARMPL_STATUS_SUCCESS) {
          std::ostringstream os;
          os << "armpl_dge_interleave(B) returned:" << info << std::endl;
          Kokkos::Impl::throw_runtime_exception(os.str());
        }

        info = armpl_dge_interleave(
            __ninter, i, __Cm, __Cn, reinterpret_cast<double *>(svC.data()),
            svC.stride(0), svC.stride(1),
            reinterpret_cast<double *>(&__Cdp[__Cbstrd * ib]), __Cistrd,
            __Cjstrd);
        if (info != ARMPL_STATUS_SUCCESS) {
          std::ostringstream os;
          os << "armpl_dge_interleave(C) returned:" << info << std::endl;
          Kokkos::Impl::throw_runtime_exception(os.str());
        }
      }
    }
  }

  void __repack_view() {
    for (int ib = 0; ib < __nbatch; ++ib) {
      for (int i = 0; i < __ninter; ++i) {
        auto svC =
            subview_wrapper(__C, ib * __ninter + i, Kokkos::ALL(),
                            Kokkos::ALL(), __batch_layout_tag, __no_trans_tag);

        auto info = armpl_dge_deinterleave(
            __ninter, i, __Cm, __Cn, reinterpret_cast<double *>(svC.data()),
            svC.stride(0), svC.stride(1),
            reinterpret_cast<double *>(&__Cdp[__Cbstrd * ib]), __Cistrd,
            __Cjstrd);
        if (info != ARMPL_STATUS_SUCCESS) {
          std::ostringstream os;
          os << "armpl_dge_deinterleave returned:" << info << std::endl;
          Kokkos::Impl::throw_runtime_exception(os.str());
        }
      }
    }
    delete __Cdp;
  }

  void __run() {
    auto info = armpl_dgemm_interleave_batch(
        __ninter, __nbatch, __transa, __transb, __Cm, __Cn,
        std::is_same<ArgTransA, Trans::NoTranspose>::value ? __An : __Am,
        static_cast<double>(__alpha), reinterpret_cast<double *>(__Adp),
        __Abstrd, __Aistrd, __Ajstrd, reinterpret_cast<double *>(__Bdp),
        __Bbstrd, __Bistrd, __Bjstrd, static_cast<double>(__beta),
        reinterpret_cast<double *>(__Cdp), __Cbstrd, __Cistrd, __Cjstrd);
    if (info != ARMPL_STATUS_SUCCESS) {
      std::ostringstream os;
      os << "armpl_dgemm_interleave_batch returned :" << info << std::endl;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
    delete __Adp;
    delete __Bdp;
  }

 public:
  BatchedArmplGemm(HandleType *const handle, ScalarType alpha, AViewType A,
                   BViewType B, ScalarType beta, CViewType C)
      : __handle(handle), __A(A), __B(B), __C(C), __alpha(alpha), __beta(beta) {
    __ninter = __handle->get_tpl_params()[0];

    if (std::is_same<ArgBatchSzDim, BatchLayout::Left>::value) {
      __Am     = __A.extent(1);
      __An     = __A.extent(2);
      __Bm     = __B.extent(1);
      __Bn     = __B.extent(2);
      __Cm     = __C.extent(1);
      __Cn     = __C.extent(2);
      __nbatch = __C.extent(0);
    } else {
      __Am     = __A.extent(0);
      __An     = __A.extent(1);
      __Bm     = __B.extent(0);
      __Bn     = __B.extent(1);
      __Cm     = __C.extent(0);
      __Cn     = __C.extent(1);
      __nbatch = __C.extent(2);
    }

    __Ajstrd = __ninter;
    __Aistrd = __Ajstrd * __An;
    __Abstrd = __Aistrd * __Am;

    __Bjstrd = __ninter;
    __Bistrd = __Bjstrd * __Bn;
    __Bbstrd = __Bistrd * __Bm;

    __Cjstrd = __ninter;
    __Cistrd = __Cjstrd * __Cn;
    __Cbstrd = __Cistrd * __Cm;

    __transa = std::is_same<ArgTransA, Trans::NoTranspose>::value ? 'N' : 'T';
    __transb = std::is_same<ArgTransB, Trans::NoTranspose>::value ? 'N' : 'T';
  }

  int invoke() {
    if (__handle->enableDebug) {
      std::cerr << "__nbatch:" << std::to_string(__nbatch)
                << ", __ninter:" << std::to_string(__ninter)
                << ", __Am:" << std::to_string(__Am)
                << ", __An:" << std::to_string(__An)
                << ", __alpha:" << std::to_string(__alpha)
                << ", __beta:" << std::to_string(__beta) << std::endl;
    }

    if (!std::is_same<avt, double>::value ||
        !std::is_same<bvt, double>::value ||
        !std::is_same<cvt, double>::value ||
        !std::is_same<ScalarType, double>::value) {
      std::ostringstream os;
      os << "KokkosBatched::Impl::BatchedArmplGemm only supports 'double' "
            "scalar types."
         << std::endl;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }

    if (__nbatch != 0) {
      if (__ninter == 0 || __nbatch % __ninter) {
        std::string msg =
            "batch size must be evenly divisible by ninter. __nbatch: ";
        msg += (std::to_string(__nbatch) +
                ", __ninter: " + std::to_string(__ninter) + "\n");
        Kokkos::abort(msg.c_str());
      }

      // Calculate internal batch size for interleaving
      __nbatch /= __ninter;

      // Allocate space for interleaving
      //   __Adp and __Bdp are deleted in __run()
      //   __Cdp is deleted in __repack_view()
      __Adp = new avt[__Abstrd * __nbatch];
      __Bdp = new bvt[__Bbstrd * __nbatch];
      __Cdp = new cvt[__Cbstrd * __nbatch];

      __unpack_views();
      __run();
      __repack_view();
    }
    return 0;
  }
};
/********************* END non-functor-level routines *********************/
}  // namespace Impl
}  // namespace KokkosBatched
#else   // KOKKOSKERNELS_ENABLE_TPL_ARMPL
namespace KokkosBatched {
namespace Impl {
/********************* BEGIN non-functor-level routines *********************/
// TODO: wrap this class in a macro for permutations of supported scalars.
template <class ArgTransA, class ArgTransB, class ArgBatchSzDim,
          class HandleType, class ScalarType, class AViewType, class BViewType,
          class CViewType>
class BatchedArmplGemm {
 public:
  BatchedArmplGemm(HandleType *const handle, ScalarType alpha, AViewType A,
                   BViewType B, ScalarType beta, CViewType C) {
    (void)handle;
    (void)alpha;
    (void)A;
    (void)B;
    (void)beta;
    (void)C;
  }

  int invoke() {
    std::ostringstream os;
    os << "KokkosBatched::Impl::BatchedArmplGemm requires the ARMPL TPL"
       << std::endl;
    Kokkos::Impl::throw_runtime_exception(os.str());
    return 1;
  }
};
}  // namespace Impl
}  // namespace KokkosBatched
#endif  // KOKKOSKERNELS_ENABLE_TPL_ARMPL
#endif
