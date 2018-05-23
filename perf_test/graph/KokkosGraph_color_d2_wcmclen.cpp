/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
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
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

// EXERCISE 1 Goal:
//   Use Kokkos to parallelize the outer loop of <y,Ax> using Kokkos::parallel_reduce.
#include <iostream>

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <Kokkos_Core.hpp>


int main( int argc, char* argv[] )
{
#if 0
    int N = 100;
    int M = 100;
    int S = 100;
    int nrepeat = 1;
    // Read command line arguments.
    for ( int i = 0; i < argc; i++ ) 
    {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-Rows" ) == 0 ) ) 
        {
            N = pow( 2, atoi( argv[ ++i ] ) );
            printf( "  User N is %d\n", N );
        }
        else if ( ( strcmp( argv[ i ], "-M" ) == 0 ) || ( strcmp( argv[ i ], "-Columns" ) == 0 ) ) 
        {
            M = pow( 2, atof( argv[ ++i ] ) );
            printf( "  User M is %d\n", M );
        }
        else if ( ( strcmp( argv[ i ], "-S" ) == 0 ) || ( strcmp( argv[ i ], "-Size" ) == 0 ) ) 
        {
            S = pow( 2, atof( argv[ ++i ] ) );
            printf( "  User S is %d\n", S );
        }
        else if ( strcmp( argv[ i ], "-nrepeat" ) == 0 ) 
        {
            nrepeat = atoi( argv[ ++i ] );
        }
        else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) 
        {
            printf( "  y^T*A*x Options:\n" );
            printf( "  -Rows (-N) <int>:      exponent num, determines number of rows 2^num (default: 2^12 = 4096)\n" );
            printf( "  -Columns (-M) <int>:   exponent num, determines number of columns 2^num (default: 2^10 = 1024)\n" );
            printf( "  -Size (-S) <int>:      exponent num, determines total matrix size 2^num (default: 2^22 = 4096*1024 )\n" );
            printf( "  -nrepeat <int>:        number of repetitions (default: 100)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
    }
#endif

    Kokkos::initialize( argc, argv );

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    std::cout << "Do stuff here!" << std::endl;

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) + 1.0e-6 * ( end.tv_usec - begin.tv_usec );

    std::cout << "Time: " << time << std::endl;

#if 0
    // Calculate bandwidth.
    // Each matrix A row (each of length M) is read once.
    // The x vector (of length M) is read N times.
    // The y vector (of length N) is read once.
    // double Gbytes = 1.0e-9 * double( sizeof(double) * ( 2 * M * N + N ) );
    double Gbytes = 1.0e-9 * double( sizeof(double) * ( M + M * N + N ) );

    // Print results (problem size, time and bandwidth in GB/s).
    printf( "  N( %d ) M( %d ) nrepeat ( %d ) problem( %g MB ) time( %g s ) bandwidth( %g GB/s )\n",
            N, M, nrepeat, Gbytes * 1000, time, Gbytes * nrepeat / time );

    delete[] A;
    delete[] y;
    delete[] x;
#endif 

  Kokkos::finalize();
  return 0;
}

