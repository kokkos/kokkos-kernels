// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_nrm2w.hpp>

int main(void) {
  Kokkos::initialize();
  {
    Kokkos::View<double*> x("X", 100), w("W", 100);
    Kokkos::deep_copy(x, 3.0);
    Kokkos::deep_copy(w, 2.0);

    double x_nrm = KokkosBlas::nrm2w(x, w);

    std::cout << "X_nrm: " << x_nrm << " Expected: " << 15 << std::endl;
  }
  Kokkos::finalize();
}
