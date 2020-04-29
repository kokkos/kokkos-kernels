/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#include <sstream>

#include <Kokkos_Core.hpp>
#include <KokkosSparse_spmv.hpp>
#include "KokkosKernels_default_types.hpp"

using Scalar    = default_scalar;
using lno_t     = default_lno_t;
using size_type = default_size_type;

void print_help() {
  printf("SPMV merge benchmark code written by Luc Berger-Vergiat.\n");
  printf("The goal is to test cusSPARSE's merge algorithm on imbalanced matrices.");
  printf("Options:\n");
  printf("  --compare       : Compare the performance of the merge algo with the default algo.\n");
  printf("  -l [LOOP]       : How many spmv to run to aggregate average time. \n");
  printf("  -numRows        : Number of rows the matrix will contain.\n");
  printf("  -numCols        : Number of columns the matrix will contain (allow rectangular matrix).\n");
  printf("  -numEntries     : Number of entries per row.\n");
  printf("  -numLongRows    : Number of rows that will contain more entries than the average.\n");
  printf("  -numLongEntries : Number of entries per row in the unbalanced rows.\n");
}

int main(int argc, char** argv) {

  bool compare        = false;
  int  loop           = 100;
  int  numRows        = 10000;
  int  numCols        = 0;
  int  numEntries     = 7;
  int  numLongRows    = 10;
  int  numLongEntries = 200;

  if(argc == 1) {
    print_help();
    return 0;
  }

  for(int i = 0; i < argc; i++) {
    if((strcmp(argv[i],"--compare"        )==0)) {compare=true; continue;}
    if((strcmp(argv[i],"-l"               )==0)) {loop=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-numRows"         )==0)) {numRows=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-numCols"         )==0)) {numCols=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-numEntries"      )==0)) {numEntries=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-numLongRows"     )==0)) {numLongRows=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"-numLongEntries"  )==0)) {numLongEntries=atoi(argv[++i]); continue;}
    if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
      print_help();
      return 0;
    }
  }

  // If numCols was not set, assume the matrix is square.
  if(numCols == 0) {numCols = numRows;}

  Kokkos::initialize(argc, argv);

  Kokkos::finalize();
} // main
