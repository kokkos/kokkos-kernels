/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
#ifndef __KOKKOSKERNELS_TEST_PARAMETERS_HPP__
#define __KOKKOSKERNELS_TEST_PARAMETERS_HPP__

namespace KokkosKernels {


namespace Example {


struct Parameters
{
    int   algorithm;
    int   repeat;
    int   chunk_size;
    int   output_graphviz_vert_max;
    int   output_graphviz;
    int   shmemsize;
    int   verbose_level;
    int   check_output;
    char* coloring_input_file;
    char* coloring_output_file;
    int   output_histogram;
    int   use_threads;
    int   use_openmp;
    int   use_cuda;
    int   use_serial;
    int   validate;
    char* mtx_bin_file;

    Parameters()
    {
        algorithm                = 0;
        repeat                   = 6;
        chunk_size               = -1;
        shmemsize                = 16128;
        verbose_level            = 0;
        check_output             = 0;
        coloring_input_file      = NULL;
        coloring_output_file     = NULL;
        output_histogram         = 0;
        output_graphviz          = 0;
        output_graphviz_vert_max = 1500;
        use_threads              = 0;
        use_openmp               = 0;
        use_cuda                 = 0;
        use_serial               = 0;
        validate                 = 0;
        mtx_bin_file             = NULL;
    }
};

}      // namespace Example

}      // namespace KokkosKernels



#endif      // __KOKKOSKERNELS_TEST_PARAMETERS_HPP__
