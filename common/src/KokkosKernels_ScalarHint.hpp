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

#ifndef KOKKOSKERNELS_SCALARHINT_HPP
#define KOKKOSKERNELS_SCALARHINT_HPP

namespace KokkosKernels::Impl {

#if 0
class ScalarHint {
public:
  enum class Kind {
    none,
    zero,
    pos_one,
    neg_one
  };

  ScalarHint(const Kind &kind) : kind_(kind) {}
  ScalarHint() : ScalarHint(Kind::none) {}

  ScalarHint(const int a) {
    if (a == -1) {
      kind_ = Kind::neg_one;
    } else if (a == 1) {
      kind_ = Kind::pos_one;
    } else {
      kind_ = Kind::none;
    }
  }

  static ScalarHint none;
  static ScalarHint zero;
  static ScalarHint pos_one;
  static ScalarHint neg_one;

private:
  Kind kind_;

};

inline ScalarHint ScalarHint::none = ScalarHint(Kind::none);
inline ScalarHint ScalarHint::zero = ScalarHint(Kind::zero);
inline ScalarHint ScalarHint::pos_one = ScalarHint(Kind::pos_one);
inline ScalarHint ScalarHint::neg_one = ScalarHint(Kind::neg_one);

#else
/*! An enum that can be used as a template param to optimize an implementation
*/
enum class ScalarHint : int {
  none,
  zero,
  pos_one,
  neg_one
};
#endif

} // namespace KokkosKernels::Impl

#endif // KOKKOSKERNELS_SCALARHINT_HPP