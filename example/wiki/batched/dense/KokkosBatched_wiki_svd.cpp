#include <iostream>
#include <Kokkos_Core.hpp>
#include <KokkosBatched_SVD_Decl.hpp>

template <class MatInfoType, class MatValuesType, class SingularValuesType, class WorkspaceType, class OffsetType>
struct SVD_functor {
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  MatInfoType mat_info;
  MatValuesType mat_values;
  SingularValuesType S;
  WorkspaceType W;
  OffsetType s_offset, w_offset;

  SVD_functor(MatInfoType mat_info_, MatValuesType mat_values_, SingularValuesType S_, WorkspaceType W_,
              OffsetType s_offset_, OffsetType w_offset_)
      : mat_info(mat_info_), mat_values(mat_values_), S(S_), W(W_), s_offset(s_offset_), w_offset(w_offset_) {}

  void KOKKOS_FUNCTION operator()(const int matIdx) const {
    Kokkos::View<double**, Kokkos::LayoutLeft, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> A(
        &mat_values(mat_info(matIdx, 0)), mat_info(matIdx, 1), mat_info(matIdx, 2));
    Kokkos::View<double*, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> s(
        &S(s_offset(matIdx)), s_offset(matIdx + 1) - s_offset(matIdx));
    Kokkos::View<double*, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> w(
        &W(w_offset(matIdx)), w_offset(matIdx + 1) - w_offset(matIdx));

    KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_S_Tag{}, A, s, w);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;

    constexpr int numMats = 2;

    // We want the following matrices in 1D storage
    // using a FORTRAN or column wise ordering.
    //
    //  A1 = [2,  1,  4]     A2 = [3,  0]
    //       [-1, 2, -2]          [4,  5]
    Kokkos::View<double*, ExecutionSpace> mat_values("values storage", 10);
    auto mat_values_h = Kokkos::create_mirror_view(mat_values);
    mat_values_h(0)   = 2;
    mat_values_h(2)   = 1;
    mat_values_h(4)   = 4;
    mat_values_h(1)   = -1;
    mat_values_h(3)   = 2;
    mat_values_h(5)   = -2;

    mat_values_h(6) = 3;
    mat_values_h(8) = 0;
    mat_values_h(7) = 4;
    mat_values_h(9) = 5;
    Kokkos::deep_copy(mat_values, mat_values_h);

    // To help our functor extract matrices from 1D storage,
    // we specify offset, numRows and numCols for each matrix.
    Kokkos::View<int**, ExecutionSpace> mat_info("matrices info", numMats, 3);
    auto mat_info_h  = Kokkos::create_mirror_view(mat_info);
    mat_info_h(0, 0) = 0;
    mat_info_h(0, 1) = 2;
    mat_info_h(0, 2) = 3;
    mat_info_h(1, 0) = 6;
    mat_info_h(1, 1) = 2;
    mat_info_h(1, 2) = 2;
    Kokkos::deep_copy(mat_info, mat_info_h);

    Kokkos::View<int*, ExecutionSpace> s_offset("s offsets", 3), w_offset("w offsets", 3);
    auto s_offset_h = Kokkos::create_mirror_view(s_offset);
    auto w_offset_h = Kokkos::create_mirror_view(w_offset);
    for (int matIdx = 0; matIdx < numMats; ++matIdx) {
      s_offset_h(matIdx + 1) = s_offset_h(matIdx) + Kokkos::min(mat_info_h(matIdx, 1), mat_info_h(matIdx, 2));
      w_offset_h(matIdx + 1) = w_offset_h(matIdx) + Kokkos::max(mat_info_h(matIdx, 1), mat_info_h(matIdx, 2));
    }
    Kokkos::deep_copy(s_offset, s_offset_h);
    Kokkos::deep_copy(w_offset, w_offset_h);

    Kokkos::View<double*, ExecutionSpace> S("singular values", 4), W("workspace", 5);

    SVD_functor svd_calculator(mat_info, mat_values, S, W, s_offset, w_offset);
    Kokkos::parallel_for(numMats, svd_calculator);

    auto S_h = Kokkos::create_mirror_view(S);
    Kokkos::deep_copy(S_h, S);
    if (Kokkos::abs(S_h(0) - 5) > 1e-14) {
      std::cout << "Large singular value of the first matrix is " << S_h(0) << "instead of 5!" << std::endl;
    }
    if (Kokkos::abs(S_h(1) - Kokkos::sqrt(5)) > 1e-14) {
      std::cout << "Large singular value of the first matrix is " << S_h(1) << "instead of " << Kokkos::sqrt(5) << "!"
                << std::endl;
    }
    if (Kokkos::abs(S_h(2) - Kokkos::sqrt(45)) > 1e-14) {
      std::cout << "Large singular value of the first matrix is " << S_h(2) << "instead of " << Kokkos::sqrt(45) << "!"
                << std::endl;
    }
    if (Kokkos::abs(S_h(3) - Kokkos::sqrt(5)) > 1e-14) {
      std::cout << "Large singular value of the first matrix is " << S_h(3) << "instead of " << Kokkos::sqrt(5) << "!"
                << std::endl;
    }

    std::cout << "Singular Values of the first  matrix: " << S_h(0) << ", " << S_h(1) << std::endl;
    std::cout << "Singular Values of the second matrix: " << S_h(2) << ", " << S_h(3) << std::endl;
  }
  Kokkos::finalize();
}
