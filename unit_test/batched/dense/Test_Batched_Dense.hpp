#ifndef TEST_BATCHED_DENSE_HPP
#define TEST_BATCHED_DENSE_HPP

// Serial kernels
#include "Test_Batched_Dense_Serial_Other.hpp"
#include "Test_Batched_Dense_Serial_BatchedGemm.hpp"

// Team Kernels
#include "Test_Batched_Dense_Team.hpp"

// TeamVector Kernels
#include "Test_Batched_Dense_TeamVector.hpp"

// Vector Kernels
#include "Test_Batched_Dense_Vector.hpp"

#endif  // TEST_BATCHED_DENSE_HPP
