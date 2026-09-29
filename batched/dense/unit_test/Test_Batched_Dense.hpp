// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef TEST_BATCHED_DENSE_HPP
#define TEST_BATCHED_DENSE_HPP

// Serial kernels
#include "Test_Batched_SerialAxpy.hpp"
#include "Test_Batched_Copy.hpp"
#include "Test_Batched_Dot.hpp"
#include "Test_Batched_Iamax.hpp"
#include "Test_Batched_Nrm.hpp"
#include "Test_Batched_Rot.hpp"
#include "Test_Batched_Rotg.hpp"
#include "Test_Batched_Rotm.hpp"
#include "Test_Batched_Rotmg.hpp"
#include "Test_Batched_Spr.hpp"
#include "Test_Batched_Swap.hpp"
#include "Test_Batched_Symv.hpp"
#include "Test_Batched_Syrk.hpp"
// NOTE: SerialEigendecomposition{,_Real} and SerialGesv{,_Real} are
// intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialInverseLU{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialLU{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
#include "Test_Batched_SerialQR.hpp"
// NOTE: SerialSolveLU{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialTrsm{,_Real,_Complex} and SerialTrmm{,_Real,_Complex} are
// intentionally NOT included here.  They have been split into standalone TUs
// under backends/Test_<Backend>_Batched_<Group>.cpp (auto-generated at
// configure time) so that `make -j` can build them in parallel with this
// umbrella.  See BATCHED_DLA_SPLIT_GROUPS in
// batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialTrsv{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialTbsv{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: SerialTrtri{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
#include "Test_Batched_SerialSVD.hpp"
#include "Test_Batched_SerialPttrf.hpp"
#include "Test_Batched_SerialPttrs.hpp"
#include "Test_Batched_SerialPbtrf.hpp"
#include "Test_Batched_SerialPbtrs.hpp"
#include "Test_Batched_SerialLaswp.hpp"
#include "Test_Batched_SerialGetrf.hpp"
#include "Test_Batched_SerialGetrs.hpp"
#include "Test_Batched_SerialGer.hpp"
#include "Test_Batched_SerialSyr.hpp"
#include "Test_Batched_SerialSyr2.hpp"
#include "Test_Batched_SerialLacgv.hpp"
#include "Test_Batched_SerialGbtrf.hpp"
#include "Test_Batched_SerialGbtrs.hpp"
#include "Test_Batched_SerialHouseholder.hpp"

// Team Kernels
#include "Test_Batched_TeamAxpy.hpp"
// NOTE: TeamGesv{,_Real} is intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: TeamInverseLU{,_Real,_Complex} and TeamLU{,_Real,_Complex} are
// intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
// NOTE: TeamSolveLU{,_Real,_Complex}, TeamTrsm{,_Real,_Complex}, and
// TeamTrsv{,_Real,_Complex} are intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.

// TeamVector Kernels
#include "Test_Batched_TeamVectorAxpy.hpp"
// NOTE: TeamVectorEigendecomposition{,_Real}, TeamVectorGesv{,_Real},
// TeamVectorQR{,_Real}, and TeamVectorQR_WithColumnPivoting{,_Real} are
// intentionally NOT included here.
// See BATCHED_DLA_SPLIT_GROUPS in batched/dense/unit_test/CMakeLists.txt.
#include "Test_Batched_TeamVectorSolveUTV.hpp"
#include "Test_Batched_TeamVectorSolveUTV_Real.hpp"
#include "Test_Batched_TeamVectorSolveUTV2.hpp"
#include "Test_Batched_TeamVectorSolveUTV2_Real.hpp"
#include "Test_Batched_TeamVectorUTV.hpp"
#include "Test_Batched_TeamVectorUTV_Real.hpp"

// Vector Kernels
#include "Test_Batched_VectorArithmatic.hpp"
#include "Test_Batched_VectorLogical.hpp"
#include "Test_Batched_VectorMath.hpp"
#include "Test_Batched_VectorMisc.hpp"
#include "Test_Batched_VectorRelation.hpp"
#include "Test_Batched_VectorView.hpp"

#endif  // TEST_BATCHED_DENSE_HPP
