// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS_CONCEPTS_HPP
#define KOKKOSBLAS_CONCEPTS_HPP

#include <concepts>
#include "KokkosBlas_util.hpp"

namespace KokkosBlas {

template <typename T>
concept TransposeOperation = is_trans_v<T>;

template <typename T>
concept BlasLevel2 = is_level2_v<T>;

template <typename T>
concept BlasLevel3 = is_level3_v<T>;

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_CONCEPTS_HPP
