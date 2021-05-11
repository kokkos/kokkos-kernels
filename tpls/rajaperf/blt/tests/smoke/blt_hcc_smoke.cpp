// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Note: This is a ROCM Hello world example from AMD:
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
//
// file: blt_hcc_smoke.cpp
//
//-----------------------------------------------------------------------------

#include <hc.hpp>
#include <hc_printf.hpp>

int main()
{
  hc::parallel_for_each(hc::extent<1>(1), []() [[hc]]
  {
    hc::printf("Accelerator: Hello World!\n");
  }).wait();

  return 0;
}


