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

/// \file Test_Common_PrintConfiguration.hpp
/// \brief Tests for print configuration

#ifndef KOKKOSKERNELS_PRINTCONFIGURATION_HPP
#define KOKKOSKERNELS_PRINTCONFIGURATION_HPP

#include "KokkosKernels_PrintConfguration.hpp"

/// \brief Verify that all keys from kernels configuration and check their value
void check_print_configuration(std::ostream& os) {
  std::ostringstream msg;
  KokkosKernels::print_configuration(msg);

  bool kernelsVersionKeyFound = false;
  bool enabledTPLsNamesKeyFound = false;
  // Iterate over lines returned from kokkos and extract key:value pairs
  std::stringstream ss{msg.str()};
  for (std::string line; std::getline(ss, line, '\n');) {
    auto found = line.find_first_of(':');
    if (found != std::string::npos) {
      auto currentKey = line.substr(0, found);
      if (currentKey == "  Kernels Version") {
          kernelsVersionKeyFound = true;
      }
      else if (currentKey == "TPLs") {
          enabledTPLsNamesKeyFound = true;
      }
    }
  }
  EXPECT_TRUE(kernelsVersionKeyFound && enabledTPLsNamesKeyFound);

}

/// \brief Verify that print_configuration print the expected keys from kernels configuration
template <typename exec_space>
void testPrintConfiguration() {
  std::ostringstream out;
  KokkosKernels::print_configuration(out);
  check_print_configuration(out);
}

TEST_F(TestCategory, common_print_configuration) { testPrintConfiguration<TestExecSpace>(); }

#endif  // KOKKOSKERNELS_PRINTCONFIGURATION_HPP
