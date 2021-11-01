#ifndef KOKKOSBLAS_IMPL_UNMQR_HPP_
#define KOKKOSBLAS_IMPL_UNMQR_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
namespace Impl {
// Put non TPL implementation here

template <class AVT, class TVT, class CVT, class WVT>
void execute_unmqr(char side, char trans, int k, AVT& A, TVT& tau, CVT& C,
                   WVT& workspace) {
  std::ostringstream os;
  os << "There is no ETI implementation of UNMQR. Compile with TPL (LAPACKE or "
        "CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
}

template <class AVT, class TVT, class CVT>
int64_t execute_unmqr_workspace(char side, char trans, int k, AVT& A, TVT& tau,
                                CVT& C) {
  std::ostringstream os;
  os << "There is no ETI implementation of UNMQR Workspace. Compile with TPL "
        "(LAPACKE or CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_IMPL_UNMQR_HPP_
