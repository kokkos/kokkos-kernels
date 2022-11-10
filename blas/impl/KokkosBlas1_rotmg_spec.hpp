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
#ifndef KOKKOSBLAS1_ROTMG_SPEC_HPP_
#define KOKKOSBLAS1_ROTMG_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas1_rotmg_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class DXView, class YView, class PView>
struct rotmg_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::Rotmg.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_ROTMG_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE,         \
                                         MEM_SPACE)                          \
  template <>                                                                \
  struct rotmg_eti_spec_avail<                                               \
      EXEC_SPACE,                                                            \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                 \
      Kokkos::View<const SCALAR, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                 \
      Kokkos::View<SCALAR[5], LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {               \
    enum : bool { value = true };                                            \
  };

// Include the actual specialization declarations
#include <KokkosBlas1_rotmg_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas1_rotmg_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

// Unification layer
template <
    class execution_space, class DXView, class YView, class PView,
    bool tpl_spec_avail =
        rotmg_tpl_spec_avail<execution_space, DXView, YView, PView>::value,
    bool eti_spec_avail =
        rotmg_eti_spec_avail<execution_space, DXView, YView, PView>::value>
struct Rotmg {
  static void rotmg(execution_space const& space, DXView& d1, DXView& d2,
                    DXView& x1, YView& y1, PView& param);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of Rotmg.
template <class execution_space, class DXView, class YView, class PView>
struct Rotmg<execution_space, DXView, YView, PView, false,
             KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void rotmg(execution_space const& space, DXView& d1, DXView& d2,
                    DXView& x1, YView& y1, PView& param) {
    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
                                      ? "KokkosBlas::rotmg[ETI]"
                                      : "KokkosBlas::rotmg[noETI]");
#ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
    if (KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
      printf("KokkosBlas1::rotmg<> ETI specialization for < %s, %s, %s >\n",
             typeid(DXView).name(), typeid(YView).name(), typeid(PView).name());
    else {
      printf("KokkosBlas1::rotmg<> non-ETI specialization for < %s, %s, %s >\n",
             typeid(DXView).name(), typeid(YView).name(), typeid(PView).name());
    }
#endif
    Rotmg_Invoke<execution_space, DXView, YView, PView>(space, d1, d2, x1, y1,
                                                        param);
    Kokkos::Profiling::popRegion();
  }
};
#endif

}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::Rotmg.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_ROTMG_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  extern template struct Rotmg<                                                \
      EXEC_SPACE,                                                              \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<const SCALAR, LAYOUT,                                       \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR[5], LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      false, true>;

//
// Macro for definition of full specialization of
// KokkosBlas::Impl::Rotmg.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define KOKKOSBLAS1_ROTMG_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct Rotmg<                                                       \
      EXEC_SPACE,                                                              \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<const SCALAR, LAYOUT,                                       \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      Kokkos::View<SCALAR[5], LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
      false, true>;

#include <KokkosBlas1_rotmg_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas1_rotmg_eti_spec_decl.hpp>

#endif  // KOKKOSBLAS1_ROTMG_SPEC_HPP_
