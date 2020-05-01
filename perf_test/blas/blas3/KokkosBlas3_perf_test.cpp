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
#include "KokkosBlas3_trmm_perf_test.hpp"

#include <cstdlib>
#include <unistd.h>
#include <getopt.h>

typedef enum TEST {
  BLAS,
  BATCHED,
  TEST_N
} test_e;

static std::string test_e_str[TEST_N] {
  "BLAS",
  "BATCHED"
};

typedef enum LOOP {
  SERIAL,
  PARALLEL,
  LOOP_N
} loop_e;

static std::string loop_e_str[LOOP_N] = {
  "SERIAL",
  "PARALLEL"
};

struct matrix_dim {
  int m, n;
};
typedef struct matrix_dim matrix_dim_t;

struct trmm_perf_test_options {
  test_e test;
  loop_e loop;
  matrix_dim_t start;
  matrix_dim_t stop;
  uint32_t step;
  uint32_t warm_up_n;
  uint32_t n;
}; 
typedef struct trmm_perf_test_options options_t;

static struct option long_options[] = {
  {"help",              no_argument,       0, 'h'},
  {"test",              required_argument, 0, 't'},
  {"loop_type",         required_argument, 0, 'l'},
  {"matrix_size_start", required_argument, 0, 'b'},
  {"matrix_size_stop",  required_argument, 0, 'e'},
  {"matrix_size_step",  required_argument, 0, 's'},
  {"warm_up_loop",      required_argument, 0, 'w'},
  {"iter",              required_argument, 0, 'i'},
  {0, 0, 0, 0}
};

#define DEFAULT_TEST BLAS
#define DEFAULT_LOOP SERIAL
#define DEFAULT_MATRIX_START 10
#define DEFAULT_MATRIX_STOP 2430
#define DEFAULT_STEP 3
#define DEFAULT_WARM_UP_N 100
#define DEFAULT_N 100

static void __print_trmm_perf_test_options(options_t options)
{
  printf("options.test      = %s\n", test_e_str[options.test].c_str());
  printf("options.loop      = %s\n", loop_e_str[options.loop].c_str());
  printf("options.start     = %dx%d\n", options.start.m, options.start.n);
  printf("options.stop      = %dx%d\n", options.stop.m, options.stop.n);
  printf("options.step      = %d\n", options.step);
  printf("options.warm_up_n = %d\n", options.warm_up_n);
  printf("options.n         = %d\n", options.n);
}
static void __print_help_trmm_perf_test()
{
  printf("Options:\n");

  printf("\t-h, --help\n");
  printf("\t\tPrint this help menu.\n\n");

  printf("\t-t, --test=OPTION\n");
  printf("\t\tAlgorithm selection.\n");
  printf("\t\t\tValid values for OPTION:\n");
  printf("%c[1m",27);
  printf("\t\t\t\tblas:");
  printf("%c[0m",27);
  printf(" invoke Kokkos::trmm the loop-body. (default)\n");
  printf("%c[1m",27);
  printf("\t\t\t\tbatched:");
  printf("%c[0m",27);
  printf(" invoke KokkosBatched::SerialTrmm in the loop-body.\n\n");

  printf("\t-l, --loop_type=OPTION\n");
  printf("\t\tLoop selection.\n");
  printf("\t\t\tValid values for OPTION:\n");
  printf("%c[1m",27);
  printf("\t\t\t\tserial:");
  printf("%c[0m",27);
  printf(" invoke trmm in a serial for-loop. (default)\n");
  printf("%c[1m",27);
  printf("\t\t\t\tparallel:");
  printf("%c[0m",27);
  printf(" invoke trmm in a Kokkos::parallel_for-loop.\n\n");

  printf("\t-b, --matrix_size_start=MxN\n");
  printf("\t\tMatrix size selection. (start)\n");
  printf("\t\t\tValid values for M and N are any non-negative 32-bit integers. (default: 10x10)\n\n");

  printf("\t-e, --matrix_size_stop=PxQ\n");
  printf("\t\tMatrix size selection. (stop)\n");
  printf("\t\t\tValid values for P and Q are any non-negative 32-bit integers. (default: 2430x2430)\n\n");
  
  printf("\t-s, --matrix_size_step=K\n");
  printf("\t\tMatrix step selection.\n");
  printf("\t\t\tValid value for K is any non-negative 32-bit integer. (default: 3)\n\n");
  
  printf("\t-w, --warm_up_loop=LOOP\n");
  printf("\t\tWarm up loop selection. (untimed)\n");
  printf("\t\t\tValid value for LOOP is any non-negative 32-bit integer. (default: 100)\n\n");
  
  printf("\t-i, --iter=ITER\n");
  printf("\t\tIteration selection. (timed)\n");
  printf("\t\t\tValid value for ITER is any non-negative 32-bit integer. (default: 100)\n\n");
}

int main(int argc, char **argv)
{
  options_t options;
  int option_idx = 0, ret;
  char *n_str = nullptr;

  /* set default options */
  options.test                                    = DEFAULT_TEST;
  options.loop                                    = DEFAULT_LOOP;
  options.start.m = options.start.n               = DEFAULT_MATRIX_START;
  options.stop.m = options.stop.n                 = DEFAULT_MATRIX_STOP;
  options.step                                    = DEFAULT_STEP;
  options.warm_up_n                               = DEFAULT_WARM_UP_N;
  options.n                                       = DEFAULT_N;

  while ((ret = getopt_long(argc, argv, "ht:l:b:e:s:w:i:", long_options, &option_idx)) != -1) {

    switch(ret) {
      case 'h':
        __print_help_trmm_perf_test();
        return 0;
      case 't':
        // printf("optarg=%s. %d\n", optarg, strncasecmp(optarg, "blas", 4));
        if (!strncasecmp(optarg, "blas", 4)) {
          options.test = BLAS;
        } else if (!strncasecmp(optarg, "batched", 6)) {
          options.test = BATCHED;
        } else {
          goto err;
        }
        break;
      case 'l':
        if (!strncasecmp(optarg, "serial", 6)) {
          options.loop = SERIAL;
        } else if (!strncasecmp(optarg, "parallel", 8)) {
          options.loop = PARALLEL;
        } else {
          goto err;
        }
        break;
      case 'b':
        n_str = strcasestr(optarg, "x");
        if (n_str == NULL)
          goto err;

        n_str[0] = '\0';
        options.start.m = atoi(optarg);
        options.start.n = atoi(&n_str[1]);
        break;
      case 'e':
        n_str = strcasestr(optarg, "x");
        if (n_str == NULL)
          goto err;

        n_str[0] = '\0';
        options.stop.m = atoi(optarg);
        options.stop.n = atoi(&n_str[1]);
        break;
      case 's':
        options.step = atoi(optarg);
        break;
      case 'w':
        options.warm_up_n = atoi(optarg);
        break;
      case 'i':
        options.n = atoi(optarg);
        break;
      case '?':
      default:
        err:
        fprintf(stderr, "ERROR: invalid option \"%s %s\".\n", argv[option_idx], argv[option_idx+1]);
        __print_help_trmm_perf_test();
        return -EINVAL;
    }
  }
  __print_trmm_perf_test_options(options);

  Kokkos::initialize(argc,argv);
  Kokkos::finalize();
  return 0;
}
