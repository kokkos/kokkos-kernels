// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "HeaderOnly.hpp"

TEST(blt_header_only_smoke,basic_assert_example)
{
  EXPECT_TRUE( blt::ReturnTrue() );
}
