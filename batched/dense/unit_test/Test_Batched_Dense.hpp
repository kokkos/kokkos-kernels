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
#include "Test_Batched_SerialQR.hpp"
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

// TeamVector Kernels
#include "Test_Batched_TeamVectorAxpy.hpp"

// Vector Kernels
#include "Test_Batched_VectorArithmatic.hpp"
#include "Test_Batched_VectorLogical.hpp"
#include "Test_Batched_VectorMath.hpp"
#include "Test_Batched_VectorMisc.hpp"
#include "Test_Batched_VectorRelation.hpp"
#include "Test_Batched_VectorView.hpp"

#endif  // TEST_BATCHED_DENSE_HPP
