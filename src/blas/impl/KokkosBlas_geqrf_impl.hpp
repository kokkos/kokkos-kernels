#ifndef KOKKOSBLAS_IMPL_GEQRF_HPP_
#define KOKKOSBLAS_IMPL_GEQRF_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
namespace Impl {
// Put non TPL implementation here

template <class AVT, class TVT, class WVT>
void execute_geqrf(AVT& A, TVT& tau, WVT& C) {
  std::ostringstream os;
  os << "There is no ETI implementation of GEQRF. Compile with TPL (LAPACKE or "
        "CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
}

template <class AVT, class TVT>
int64_t execute_geqrf_workspace(AVT& A, TVT& tau) {
  std::ostringstream os;
  os << "There is no ETI implementation of GEQRF (Workspace Query). Compile "
        "with TPL (LAPACKE or CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_IMPL_GEQRF_HPP_
