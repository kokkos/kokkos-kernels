//
// Created by Poliakoff, David Zoeller on 4/27/21.
//

#ifndef KOKKOSKERNELS_PERFTESTUTILITIES_HPP
#define KOKKOSKERNELS_PERFTESTUTILITIES_HPP
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_config.h"
#include "KokkosKernels_IOUtils.hpp"
#include <common/RunParams.hpp>
#include <common/QuickKernelBase.hpp>
#include <common/KernelBase.hpp>
#include <common/QuickKernelBase.hpp>
#include <common/KernelBase.hpp>
#include <dirent.h>


namespace test {

 std::string inputDataPath;

 void set_input_data_path(const std::string& path_to_data){
   inputDataPath = path_to_data;	

};
 std::string get_input_data_path(){
  	
return inputDataPath;

};
}

#endif  // KOKKOSKERNELS_PERFTESTUTILITIES_HPP
