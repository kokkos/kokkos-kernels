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

#ifndef KOKKOS_HALFPRECISION_HPP
#define KOKKOS_HALFPRECISION_HPP

#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_KOKKOSKERNELS_CUDA_FP16)
#include <cuda_fp16.h>
#endif

namespace KokkosKernels {
    namespace Experimental {
        /**
         * Below we check whether the given toolchain has support for portable IEEE-754
         * FP16 (binary16) precision types. The checks are done via CMake which passes the
         * results via a KOKKOSKERNELS_HAVE define to KokkosKernels_config.h
         * 
         * First we check for cuda half precision support   (HAVE_KOKKOSKERNELS_CUDA_FP16).
         * Second we check for host half precision support  (HAVE_KOKKOSKERNELS_FP16).
         * Lastly, we fall back to single precision support.
         * 
         * NOTE: If both cuda and host support half precision, the half type will
         * default to device_fp16_t.
         */
        #if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_KOKKOSKERNELS_CUDA_FP16)
            using device_fp16_t = __half;
            using half = device_fp16_t;
            #if defined(HAVE_KOKKOSKERNELS_FP16)
                using host_fp16_t = _Float16;
            #else
                using host_fp16_t = float;
            #endif // defined(HAVE_KOKKOSKERNELS_FP16)
            static KOKKOS_FORCEINLINE_FUNCTION float __cast2float(device_fp16_t x) { return __half2float(x); }
            static KOKKOS_FORCEINLINE_FUNCTION device_fp16_t __cast2half(float x) { return __float2half(x); }
            
        #else // defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_KOKKOSKERNELS_CUDA_FP16)
            #if defined(HAVE_KOKKOSKERNELS_FP16)
                using host_fp16_t = _Float16;
                using half = host_fp16_t;
            #else // defined(HAVE_KOKKOSKERNELS_FP16)
                using host_fp16_t = float;
                using device_fp16_t = host_fp16_t;
                using half = host_fp16_t;
            #endif // _Float16
            static inline float __cast2float(host_fp16_t x) { return (float) x; }
            static inline host_fp16_t __cast2half(float x) { return (host_fp16_t) x; }
        #endif
        ////////////// BEGIN half2float and float2half overloads //////////////
        /**
        * Since kokkos does not have support for half precision types yet, we 
        * must cast to/from float in some kokkos-kernels routines. Except for
        * the overloads below that actually cast to/from half precision types,
        * the others should be optimized away by the compiler.
        */
        // host_fp16_t
#if defined(HAVE_KOKKOSKERNELS_FP16)
        static inline
        float half2float(host_fp16_t x, float &ret) {
            ret = __cast2float(x);
            return ret;
        }
        static inline
        host_fp16_t float2half(float x, host_fp16_t &ret) {
            ret = __cast2half(x);
            return ret;
        }
#endif
        // device_fp16_t
#if defined(KOKKOS_ENABLE_CUDA) && defined(HAVE_KOKKOSKERNELS_CUDA_FP16)
        static inline __device__
        float half2float(device_fp16_t x, float &ret) {
            ret = __cast2float(x);
            return ret;
        }
        static inline __device__
        device_fp16_t float2half(float x, device_fp16_t &ret) {
            ret = __cast2half(x);
            return ret;
        }
#endif
        // float
        static KOKKOS_FORCEINLINE_FUNCTION float half2float(float x, float &ret) {
            ret = x;
            return ret;
        }
        static KOKKOS_FORCEINLINE_FUNCTION float float2half(float x, float &ret) {
            ret = x;
            return ret;
        }
        // complex float
        static KOKKOS_FORCEINLINE_FUNCTION Kokkos::complex<float> half2float(Kokkos::complex<float> x, Kokkos::complex<float> &ret) {
            ret = x;
            return ret;
        }
        static KOKKOS_FORCEINLINE_FUNCTION Kokkos::complex<float> float2half(Kokkos::complex<float> x, Kokkos::complex<float> &ret) {
            ret = x;
            return ret;
        }
        // double
        static KOKKOS_FORCEINLINE_FUNCTION double half2float(double x, double &ret) {
            ret = x;
            return ret;
        }
        static KOKKOS_FORCEINLINE_FUNCTION double float2half(double x, double &ret) {
            ret = x;
            return ret;
        }
        // complex double
        static KOKKOS_FORCEINLINE_FUNCTION Kokkos::complex<double> half2float(Kokkos::complex<double> x, Kokkos::complex<double> &ret) {
            ret = x;
            return ret;
        }
        static KOKKOS_FORCEINLINE_FUNCTION Kokkos::complex<double> float2half(Kokkos::complex<double> x, Kokkos::complex<double> &ret) {
            ret = x;
            return ret;
        }
        ////////////// END half2float and float2half overloads //////////////

        ////////////// BEGIN FP16/binary16 limits //////////////
        #define FP16_MAX 65504.0F           // Maximum normalized number
        #define FP16_MIN 0.000000059604645F // Minimum normalized positive half precision number
        #define FP16_RADIX 2                // Value of the base of the exponent representation. TODO: Confirm this
        #define FP16_MANT_DIG 15            // Number of digits in the matissa that can be represented without losing precision. TODO: Confirm this
        #define FP16_MIN_EXP -14            // This is the smallest possible exponent value
        #define FP16_MAX_EXP 15             // This is the largest possible exponent value
        #define FP16_SIGNIFICAND_BITS 10
        #define FP16_EPSILON 0.0009765625F
        #define HUGE_VALH 0x7c00            // bits [10,14] set.
        ////////////// END FP16/binary16 limits //////////////
    } // Experimental
} // KokkosKernels
#endif // KOKKOS_HALFPRECISION_HPP
