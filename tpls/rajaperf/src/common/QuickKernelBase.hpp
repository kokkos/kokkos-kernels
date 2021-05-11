#ifndef RAJAPERFSUITE_QUICKKERNELBASE_HPP
#define RAJAPERFSUITE_QUICKKERNELBASE_HPP

#include "KernelBase.hpp"
#include <utility>

namespace rajaperf {

struct SureBuddyOkay {
  bool validate_checksum(double reference, double variant) { return true; }
};

template <typename SetUp, typename Execute, typename Checksum = SureBuddyOkay>
class QuickKernelBase : public rajaperf::KernelBase {
  SetUp m_setup;
  Execute m_execute;
  Checksum m_checksum;
  struct empty {};
  using runData_helper = decltype(m_setup(0, 0));
  using runData =
      typename std::conditional<std::is_same<runData_helper, void>::value,
                                empty, runData_helper>::type;
  using is_empty = std::is_same<runData, empty>;
  runData* rd;

 public:
  // Index_type getDefaultSize() const { return 1000; }
  // Index_type getDefaultReps() const { return 1000; }
  // Index_type getItsPerRep() const override { return 1000; }
  // Index_type getRunReps() const { return 1; }
  QuickKernelBase(std::string &name, const RunParams &params,
                  SetUp setup_lambda, Execute test_lambda,
                  Checksum checksum_lambda)
      : KernelBase(name, params),
        m_setup(setup_lambda),
        m_execute(test_lambda),
        m_checksum(checksum_lambda) {
    setVariantDefined(Kokkos_Lambda);
    setDefaultSize(100000);
    setDefaultReps(5000);
  }

  QuickKernelBase(std::string &name, const RunParams &params,
                  SetUp setup_lambda, Execute test_lambda)
      : KernelBase(name, params),
        m_setup(setup_lambda),
        m_execute(test_lambda),
        m_checksum(SureBuddyOkay()) {
    setVariantDefined(Kokkos_Lambda);
    setDefaultSize(100000);
    setDefaultReps(50);
  }

  Real_type m_y;

  void setUpHelper(std::true_type) { m_setup(getItsPerRep(), getRunSize()); }

  void setUpHelper(std::false_type) {
    rd = new runData(m_setup(getItsPerRep(), getRunSize()));
  }

  void setUp(VariantID vid) override { setUpHelper(is_empty()); }

  void updateChecksum(VariantID vid) override { checksum[vid] += m_y; }

  void tearDown(VariantID vID) override {}

  void runSeqVariant(VariantID vID) override {}

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  void runOpenMPVariant(VariantID vid) override {
    auto size = getRunSize();
    for (int x = 0; x < getRunReps(); ++x) {
      m_execute(x, size)
    }
  }
#endif
#if defined(RAJA_ENABLE_CUDA)
  void runCudaVariant(VariantID vid) override {}
#endif
#if defined(RAJA_ENABLE_HIP)
  void runHipVariant(VariantID vid) override {}
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
  void runOpenMPTargetVariant(VariantID vid) override {}
#endif

#if defined(RUN_KOKKOS) or defined(RAJAPERF_INFRASTRUCTURE_ONLY)

  template <size_t... Is>
  void rkv_helper(std::index_sequence<Is...>) {
    auto size = getRunSize();
    for (int x = 0; x < getRunReps(); ++x) {
      m_execute(x, size, std::get<Is>(*rd)...);
    }
  }

  void rkv_helper(empty em) {
    auto size = getRunSize();
    for (int x = 0; x < getDefaultSize(); ++x) {
      m_execute(x, size);
    }
  }

  void rkv_switch_on_empty(std::false_type) {
    using index_seq =
        typename std::make_index_sequence<std::tuple_size<runData>::value>;
    rkv_helper(index_seq());
  }

  void rkv_switch_on_empty(std::true_type) { rkv_helper(empty()); }

  void runKokkosVariant(VariantID vid) override {
    startTimer();
    rkv_switch_on_empty(is_empty());
    stopTimer();
  }

#endif  // RUN_KOKKOS
  ~QuickKernelBase() { free(rd);}
};

template <class... Lambdas>
KernelBase *make_kernel_base(std::string name, const RunParams &params,
                             Lambdas... lambdas) {
  return new QuickKernelBase<Lambdas...>(name, params, lambdas...);
}

}  // end namespace rajaperf
#endif  // RAJAPERFSUITE_QUICKKERNELBASE_HPP
