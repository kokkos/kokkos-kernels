// Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
// other BLT Project Developers. See the top-level COPYRIGHT file for details
//
// SPDX-License-Identifier: (BSD-3-Clause)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Note: This is a ROCm example from AMD:
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
//
// file: blt_hcc_runtime_smoke.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include "hc.hpp"

int main()
{
  using namespace hc;
  accelerator acc;
  std::vector<accelerator> accv = acc.get_all() ;

  std::cout << "Found " << accv.size() << " accelerators."  << std::endl;
  std::cout << std::endl;

  unsigned long idefault  = 0;
  for(unsigned long i=0; i< accv.size(); i++)
  {
    accelerator a = accv[i];
    std::cout << "Accelerator " << i << ": " ;
    std::wcout << a.get_device_path() << L" : " <<  a.get_description();
    std::cout <<  " : " <<  (a.get_version()>>16) <<  "."
              << (a.get_version()&0xff);
    std::cout << std::endl;
    if (a == acc) { idefault = i; }
  }

  std::cout <<  std::endl;
  std::cout << "Default Accelerator " << ": " << idefault << " : " ;
  std::wcout << acc.get_device_path() <<  std::endl;

  return 0;
}

