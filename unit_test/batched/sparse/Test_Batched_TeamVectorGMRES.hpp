/// \author Kim Liegeois (knliege@sandia.gov)

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "KokkosBatched_GMRES.hpp"
#include "KokkosKernels_TestUtils.hpp"
#include "KokkosBatched_CrsMatrix.hpp"
#include "Test_Batched_SparseUtils.hpp"
#include "KokkosBatched_JacobiPrec.hpp"

using namespace KokkosBatched;

namespace Test {
namespace TeamVectorGMRES {

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType>
struct Functor_TestBatchedTeamVectorGMRES {
  const ValuesViewType _D;
  const ValuesViewType _Diag;
  const IntView _r;
  const IntView _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int _N_team;
  KrylovHandle<typename ValuesViewType::value_type> handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const IntView &r,
                                     const IntView &c, const VectorViewType &X,
                                     const VectorViewType &B,
                                     const VectorViewType &diag,
                                     const int N_team)
      : _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _Diag(diag) {}

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const {
    const int first_matrix = static_cast<int>(member.league_rank()) * _N_team;
    const int N            = _D.extent(0);
    const int last_matrix =
        (static_cast<int>(member.league_rank() + 1) * _N_team < N
             ? static_cast<int>(member.league_rank() + 1) * _N_team
             : N);

    auto d = Kokkos::subview(_D, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto diag = Kokkos::subview(
        _Diag, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    auto x = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);
    auto b = Kokkos::subview(_B, Kokkos::make_pair(first_matrix, last_matrix),
                             Kokkos::ALL);

    using Operator     = KokkosBatched::CrsMatrix<ValuesViewType, IntView>;
    using PrecOperator = KokkosBatched::JacobiPrec<ValuesViewType>;

    Operator A(d, _r, _c);
    PrecOperator P(diag);

    KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator,
                                                                VectorViewType>(
        member, A, b, x, handle, P);
  }

  inline void run() {
    typedef typename ValuesViewType::value_type value_type;
    std::string name_region("KokkosBatched::Test::TeamVectorGMRES");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::TeamPolicy<DeviceType> policy(_D.extent(0) / _N_team,
                                          Kokkos::AUTO(), Kokkos::AUTO());

    size_t bytes_0 = ValuesViewType::shmem_size(_N_team, _D.extent(1));
    size_t bytes_1 = ValuesViewType::shmem_size(_N_team, 1);

    handle.set_max_iteration(10);

    int maximum_iteration = handle.get_max_iteration();

    policy.set_scratch_size(0, Kokkos::PerTeam(5 * bytes_0 + 5 * bytes_1));
    policy.set_scratch_size(
        1, Kokkos::PerTeam(maximum_iteration * bytes_0 +
                           ((maximum_iteration + 3) * maximum_iteration) *
                               bytes_1));

    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ValuesViewType, typename IntView,
          typename VectorViewType>
void impl_test_batched_GMRES(const int N, const int BlkSize, const int N_team) {
  typedef typename ValuesViewType::value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type> ats;

  const int nnz = (BlkSize - 2) * 3 + 2 * 2;

  VectorViewType X("x0", N, BlkSize);
  VectorViewType R("r0", N, BlkSize);
  VectorViewType B("b", N, BlkSize);
  ValuesViewType D("D", N, nnz);
  ValuesViewType Diag("Diag", N, BlkSize);
  IntView r("r", BlkSize + 1);
  IntView c("c", nnz);

  using ScalarType = typename ValuesViewType::non_const_value_type;
  using Layout     = typename ValuesViewType::array_layout;
  using EXSP       = typename ValuesViewType::execution_space;

  using MagnitudeType =
      typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
  using NormViewType = Kokkos::View<MagnitudeType *, Layout, EXSP>;

  NormViewType sqr_norm_0("sqr_norm_0", N);
  NormViewType sqr_norm_j("sqr_norm_j", N);

  create_tridiagonal_batched_matrices(nnz, BlkSize, N, r, c, D, X, B);

  {
    auto diag_values_host = Kokkos::create_mirror_view(Diag);
    auto values_host      = Kokkos::create_mirror_view(D);
    auto row_ptr_host     = Kokkos::create_mirror_view(r);
    auto colIndices_host  = Kokkos::create_mirror_view(c);

    Kokkos::deep_copy(values_host, D);
    Kokkos::deep_copy(row_ptr_host, r);
    Kokkos::deep_copy(colIndices_host, c);

    int current_index;
    for (int i = 0; i < BlkSize; ++i) {
      for (current_index = row_ptr_host(i); current_index < row_ptr_host(i + 1);
           ++current_index) {
        if (colIndices_host(current_index) == i) break;
      }
      for (int j = 0; j < N; ++j)
        diag_values_host(j, i) = values_host(j, current_index);
    }

    Kokkos::deep_copy(Diag, diag_values_host);
  }

  // Compute initial norm

  Kokkos::deep_copy(R, B);

  auto sqr_norm_0_host = Kokkos::create_mirror_view(sqr_norm_0);
  auto sqr_norm_j_host = Kokkos::create_mirror_view(sqr_norm_j);
  auto R_host          = Kokkos::create_mirror_view(R);
  auto X_host          = Kokkos::create_mirror_view(X);
  auto D_host          = Kokkos::create_mirror_view(D);
  auto r_host          = Kokkos::create_mirror_view(r);
  auto c_host          = Kokkos::create_mirror_view(c);

  Kokkos::deep_copy(R, B);
  Kokkos::deep_copy(R_host, R);
  Kokkos::deep_copy(X_host, X);

  Kokkos::deep_copy(c_host, c);
  Kokkos::deep_copy(r_host, r);
  Kokkos::deep_copy(D_host, D);

  KokkosBatched::SerialSpmv<Trans::NoTranspose>::template invoke<
      typename ValuesViewType::HostMirror, typename IntView::HostMirror,
      typename VectorViewType::HostMirror, typename VectorViewType::HostMirror,
      1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
  KokkosBatched::SerialDot<Trans::NoTranspose>::invoke(R_host, R_host,
                                                       sqr_norm_0_host);
  Functor_TestBatchedTeamVectorGMRES<DeviceType, ValuesViewType, IntView,
                                     VectorViewType>(D, r, c, X, B, Diag,
                                                     N_team)
      .run();

  Kokkos::fence();

  Kokkos::deep_copy(R, B);
  Kokkos::deep_copy(R_host, R);
  Kokkos::deep_copy(X_host, X);

  KokkosBatched::SerialSpmv<Trans::NoTranspose>::template invoke<
      typename ValuesViewType::HostMirror, typename IntView::HostMirror,
      typename VectorViewType::HostMirror, typename VectorViewType::HostMirror,
      1>(-1, D_host, r_host, c_host, X_host, 1, R_host);
  KokkosBatched::SerialDot<Trans::NoTranspose>::invoke(R_host, R_host,
                                                       sqr_norm_j_host);

  const MagnitudeType eps = 1.0e5 * ats::epsilon();

  for (int l = 0; l < N; ++l)
    EXPECT_NEAR_KK(
        std::sqrt(sqr_norm_j_host(l)) / std::sqrt(sqr_norm_0_host(l)), 0, eps);
}
}  // namespace TeamVectorGMRES
}  // namespace Test

template <typename DeviceType, typename ValueType>
int test_batched_teamvector_GMRES() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    typedef Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType> ViewType;
    typedef Kokkos::View<int *, Kokkos::LayoutLeft, DeviceType> IntView;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutLeft, DeviceType>
        VectorViewType;

    for (int i = 3; i < 10; ++i) {
      Test::TeamVectorGMRES::impl_test_batched_GMRES<DeviceType, ViewType,
                                                     IntView, VectorViewType>(
          1024, i, 2);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        ViewType;
    typedef Kokkos::View<int *, Kokkos::LayoutRight, DeviceType> IntView;
    typedef Kokkos::View<ValueType **, Kokkos::LayoutRight, DeviceType>
        VectorViewType;

    for (int i = 3; i < 10; ++i) {
      Test::TeamVectorGMRES::impl_test_batched_GMRES<DeviceType, ViewType,
                                                     IntView, VectorViewType>(
          1024, i, 2);
    }
  }
#endif

  return 0;
}
