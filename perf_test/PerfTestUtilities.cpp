//
// Created by Poliakoff, David Zoeller on 4/27/21.
//
/*
#include "KokkosKernels_default_types.hpp"
#include "KokkosKernels_config.h"
#include "KokkosKernels_IOUtils.hpp"
#include <common/RunParams.hpp>
#include <common/QuickKernelBase.hpp>
#include <common/KernelBase.hpp>
#include <dirent.h>
*/

#include<string>

namespace test {

std::string inputDataPath;

void set_input_data_path(const std::string& path_to_data) {
  inputDataPath = path_to_data;
};
std::string get_input_data_path() { return inputDataPath; };
}  // namespace test
