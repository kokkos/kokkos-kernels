//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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

template <class XType>
void write2DArrayToMM(std::string name, const XType x) {
  std::ofstream myfile;
  myfile.open(name);

  auto x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  if (XType::Rank == 2) {
    myfile << "%% MatrixMarket 2D Array\n%" << std::endl;
    myfile << x_h.extent(0) << " " << x_h.extent(1) << std::endl;

    for (size_t i = 0; i < x_h.extent(0); ++i) {
      for (size_t j = 0; j < x_h.extent(1); ++j) {
        myfile << std::setprecision(15) << x_h(i, j) << " ";
      }
      myfile << std::endl;
    }

    myfile.close();
  }
}

template <class XType>
void write3DArrayToMM(std::string name, const XType x) {
  std::ofstream myfile;
  myfile.open(name);

  auto x_h = Kokkos::create_mirror_view(x);

  Kokkos::deep_copy(x_h, x);

  if (XType::Rank == 3) {
    myfile << "%% MatrixMarket 3D Array\n%" << std::endl;
    myfile << x_h.extent(0) << " " << x_h.extent(1) << " " << x_h.extent(2)
           << std::endl;

    for (size_t i = 0; i < x_h.extent(0); ++i) {
      myfile << "Slice " << i << std::endl;
      for (size_t j = 0; j < x_h.extent(1); ++j) {
        for (size_t k = 0; k < x_h.extent(2); ++k) {
          myfile << std::setprecision(15) << x_h(i, j, k) << " ";
        }
        myfile << std::endl;
      }
    }

    myfile.close();
  }
}

template <typename MatrixViewType, typename VectorViewType>
void create_saddle_point_matrices(const MatrixViewType &A,
                                  const VectorViewType &Y, const int n_2 = 4) {
  Kokkos::Random_XorShift64_Pool<
      typename MatrixViewType::device_type::execution_space>
      random(13718);
  const int N   = A.extent(0);
  const int n   = A.extent(1);
  const int n_1 = n - n_2;

  const int n_dim = n_2 - 1;
  MatrixViewType xs("xs", N, n_1, n_dim);
  VectorViewType ys("ys", N, n_1);

  Kokkos::fill_random(
      xs, random,
      Kokkos::reduction_identity<typename MatrixViewType::value_type>::prod());
  Kokkos::fill_random(
      ys, random,
      Kokkos::reduction_identity<typename VectorViewType::value_type>::prod());

  auto xs_host = Kokkos::create_mirror_view(xs);
  auto ys_host = Kokkos::create_mirror_view(ys);
  auto A_host  = Kokkos::create_mirror_view(A);
  auto Y_host  = Kokkos::create_mirror_view(Y);

  Kokkos::deep_copy(xs_host, xs);
  Kokkos::deep_copy(ys_host, ys);

  for (int i = 0; i < n_1; ++i) {
    for (int j = 0; j < n_1; ++j) {
      for (int l = 0; l < N; ++l) {
        auto xs_i = Kokkos::subview(xs_host, l, i, Kokkos::ALL);
        auto xs_j = Kokkos::subview(xs_host, l, j, Kokkos::ALL);
        typename MatrixViewType::value_type d = 0;
        for (int k = 0; k < n_dim; ++k) d += Kokkos::pow(xs_i(k) - xs_j(k), 2);
        d               = Kokkos::sqrt(d);
        A_host(l, i, j) = Kokkos::pow(d, 5);
      }
    }
    for (int l = 0; l < N; ++l) {
      A_host(l, i, n_1) = (typename MatrixViewType::value_type)1.0;
      A_host(l, n_1, i) = (typename MatrixViewType::value_type)1.0;
      for (int k = 0; k < n_dim; ++k) {
        A_host(l, i, n_1 + k + 1) = xs_host(l, i, k);
        A_host(l, n_1 + k + 1, i) = xs_host(l, i, k);
      }
      Y_host(l, i) = ys_host(l, i);
    }
  }
  for (int i = n_1; i < n; ++i) {
    for (int l = 0; l < N; ++l) {
      Y_host(l, i) = (typename MatrixViewType::value_type)0.0;
    }
  }

  Kokkos::deep_copy(A, A_host);
  Kokkos::deep_copy(Y, Y_host);

  Kokkos::fence();
}