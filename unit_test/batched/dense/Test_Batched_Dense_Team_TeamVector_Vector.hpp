#ifndef TEST_BATCHED_DENSE_TEAM_TEAMVECTOR_VECTOR_HPP
#define TEST_BATCHED_DENSE_TEAM_TEAMVECTOR_VECTOR_HPP

// for CUDA backend the batched dense tests are split into multiple TUs, and
// this file may be used alone to define a set of tests need to bring some
// definitions in
#include "Test_Batched_SerialGemv.hpp"       // test_batched_gemv
#include "Test_Batched_SerialInverseLU.hpp"  // test_batched_inverselu
#include "Test_Batched_SerialLU.hpp"         // test_batched_lu
#include "Test_Batched_SerialMatUtil.hpp"    // BatchedSet

// Team Kernels
#include "Test_Batched_TeamAxpy.hpp"
#include "Test_Batched_TeamAxpy_Real.hpp"
#include "Test_Batched_TeamAxpy_Complex.hpp"
#include "Test_Batched_TeamGemm.hpp"
#include "Test_Batched_TeamGemm_Real.hpp"
#include "Test_Batched_TeamGemm_Complex.hpp"
#include "Test_Batched_TeamGemv.hpp"
#include "Test_Batched_TeamGemv_Real.hpp"
#include "Test_Batched_TeamGemv_Complex.hpp"
#include "Test_Batched_TeamGesv.hpp"
#include "Test_Batched_TeamGesv_Real.hpp"
#include "Test_Batched_TeamInverseLU.hpp"
#include "Test_Batched_TeamInverseLU_Real.hpp"
#include "Test_Batched_TeamInverseLU_Complex.hpp"
#include "Test_Batched_TeamLU.hpp"
#include "Test_Batched_TeamLU_Real.hpp"
#include "Test_Batched_TeamLU_Complex.hpp"
#include "Test_Batched_TeamMatUtil.hpp"
#include "Test_Batched_TeamMatUtil_Real.hpp"
#include "Test_Batched_TeamMatUtil_Complex.hpp"
#include "Test_Batched_TeamSolveLU.hpp"
#include "Test_Batched_TeamSolveLU_Real.hpp"
#include "Test_Batched_TeamSolveLU_Complex.hpp"
#include "Test_Batched_TeamTrsm.hpp"
#include "Test_Batched_TeamTrsm_Real.hpp"
#include "Test_Batched_TeamTrsm_Complex.hpp"
#include "Test_Batched_TeamTrsv.hpp"
#include "Test_Batched_TeamTrsv_Real.hpp"
#include "Test_Batched_TeamTrsv_Complex.hpp"

// TeamVector Kernels
#include "Test_Batched_TeamVectorAxpy.hpp"
#include "Test_Batched_TeamVectorAxpy_Real.hpp"
#include "Test_Batched_TeamVectorAxpy_Complex.hpp"
#include "Test_Batched_TeamVectorEigendecomposition.hpp"
#include "Test_Batched_TeamVectorEigendecomposition_Real.hpp"
#include "Test_Batched_TeamVectorGemm.hpp"
#include "Test_Batched_TeamVectorGemm_Real.hpp"
#include "Test_Batched_TeamVectorGemm_Complex.hpp"
#include "Test_Batched_TeamVectorGesv.hpp"
#include "Test_Batched_TeamVectorGesv_Real.hpp"
#include "Test_Batched_TeamVectorQR.hpp"
#include "Test_Batched_TeamVectorQR_Real.hpp"
#include "Test_Batched_TeamVectorQR_WithColumnPivoting.hpp"
#include "Test_Batched_TeamVectorQR_WithColumnPivoting_Real.hpp"
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

#endif  // TEST_BATCHED_DENSE_TEAM_TEAMVECTOR_VECTOR_HPP
