/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Kokkos_Core.hpp"
#include "impl/Kokkos_Timer.hpp"

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

using namespace KokkosBatched::Experimental;

void Test() {
  static constexpr int vl = 4;
  typedef double value_type;
  typedef Vector<SIMD<value_type>,vl> vector_type;
  typedef Vector<SIMD<bool>,vl> vector_bool_type;
  typedef Vector<SIMD<int>,vl> vector_int_type;

  ///
  /// Math function overload
  ///
  Random<value_type> random;

  {
    vector_type a, b, ref, aref, bref;
    for (int i=0;i<vl;++i) {
      ref[i] = (random.value() + 1.0)/2;
      aref[i] = (random.value() + 1.0)/2;
      bref[i] = (random.value() + 1.0)/2;
    }

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      a = op(ref); b = ref;                                             \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::isnan(a[i])*1000;                                  \
        diff += std::abs(a[i] - std::op(b[i]));                         \
      }                                                                 \
      std::cout << #op << "[vector] diff = " << diff << "\n"; \
    }
    
    CHECK(sqrt);
    CHECK(cbrt);
    CHECK(log);
    CHECK(exp);
    CHECK(sin);
    CHECK(cos);
    CHECK(tan);
    CHECK(sinh);
    CHECK(cosh);
    CHECK(tanh);
    CHECK(asin);
    CHECK(acos);
    CHECK(atan);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      a = op(aref,bref);                                                \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::isnan(a[i]);                                     \
        diff += std::abs(a[i] - std::op(aref[i],bref[i]));              \
      }                                                                 \
      std::cout << #op << "[vector,vector] diff = " << diff << "\n"; \
    }
    
    CHECK(pow);
    CHECK(atan2);
    
#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type beta = random.value();                                 \
      value_type diff = 0;                                              \
      a = op(aref,beta);                                                \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::isnan(a[i]);                                     \
        diff += std::abs(a[i] - std::op(aref[i],beta));                 \
      }                                                                 \
      std::cout << #op << "[vector,scalar] diff = " << diff << "\n"; \
    }
    
    CHECK(pow);
    CHECK(atan2);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type alpha = random.value();                                \
      value_type diff = 0;                                              \
      a = op(alpha,bref);                                               \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::isnan(a[i]);                                     \
        diff += std::abs(a[i] - std::op(alpha,bref[i]));                \
      }                                                                 \
      std::cout << #op << "[scalar,vector] diff = " << diff << "\n"; \
    }
    
    CHECK(pow);
    CHECK(atan2);
  }


  ///
  /// Relation operator
  ///
  {
    vector_type a, b;
    a[0] = 1.0; b[0] = 1.0;
    for (int i=1;i<vl;++i) {
      a[i] = random.value();
      b[i] = random.value();
    }
    
#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = a op b;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (a[i] op b[i]));                     \
      }                                                                 \
      std::cout << #op << "[vector,vector] diff = " << diff << "\n";    \
    }
    
    CHECK(<);
    CHECK(>);
    CHECK(>);
    CHECK(<=);
    CHECK(>=);
    CHECK(==);
    CHECK(!=);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = a op 0;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (a[i] op 0));                        \
      }                                                                 \
      std::cout << #op << "[vector,scalar] diff = " << diff << "\n";    \
    }
    
    CHECK(<);
    CHECK(>);
    CHECK(>);
    CHECK(<=);
    CHECK(>=);
    CHECK(==);
    CHECK(!=);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = 0 op b;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (0 op b[i]));                        \
      }                                                                 \
      std::cout << #op << "[scalar,vector] diff = " << diff << "\n";    \
    }
    
    CHECK(<);
    CHECK(>);
    CHECK(>);
    CHECK(<=);
    CHECK(>=);
    CHECK(==);
    CHECK(!=);
  }


  ///
  /// Logical operator
  ///
  {
    vector_int_type a, b;
    for (int i=1;i<vl;++i) {
      a[i] = (random.value() > 0 ? 1 : -1) * 4;
      b[i] = (random.value() < 0 ? 1 : -1) * 3;
    }
    
#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = a op b;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (a[i] op b[i]));                     \
      }                                                                 \
      std::cout << #op << "[vector,vector] diff = " << diff << "\n";    \
    }
    
    CHECK(||);
    CHECK(&&);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = a op 0;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (a[i] op 0));                        \
      }                                                                 \
      std::cout << #op << "[vector,scalar] diff = " << diff << "\n";    \
    }
    
    CHECK(||);
    CHECK(&&);

#undef CHECK
#define CHECK(op)                                                       \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = 0 op b;                                         \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - (0 op b[i]));                        \
      }                                                                 \
      std::cout << #op << "[scalar,vector] diff = " << diff << "\n";    \
    }
    
    CHECK(||);
    CHECK(&&);

#undef CHECK
#define CHECK                                                           \
    {                                                                   \
      value_type diff = 0;                                              \
      const auto comp = !a;                                             \
      for (int i=0;i<vl;++i) {                                          \
        diff += std::abs(comp[i] - !b[i]);                              \
      }                                                                 \
      std::cout << "!" << "[vector] diff = " << diff << "\n";    \
    }
    
    CHECK;
  }


  ///
  /// Misc
  ///
  {
    vector_type a, b, c;
    for (int i=0;i<vl;++i) {
      a[i] = random.value();
      b[i] = random.value();
    }

    {
      value_type diff = 0;
      c = conditional_assign(a < b, a, b);
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? a[i] : b[i];
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[vector,vector] diff = " << diff << "\n";

      c = 0; conditional_assign(c, a < b, a, b);
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? a[i] : b[i];
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[vector,vector] diff = " << diff << "\n";
    }
    {
      value_type diff = 0;
      c = conditional_assign(a < b, a, value_type(0));
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? a[i] : 0;
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[vector,scalar] diff = " << diff << "\n";

      c = 0; conditional_assign(c, a < b, a, value_type(0));
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? a[i] : 0;
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[vector,scalar] diff = " << diff << "\n";
    }
    {
      value_type diff = 0;
      c = conditional_assign(a < b, value_type(0), b);
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? 0 : b[i];
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[scalar,vector] diff = " << diff << "\n";

      c = 0; conditional_assign(c, a < b, value_type(0), b);
      for (int i=0;i<vl;++i) {
        const auto cc = a[i] < b[i] ? 0 : b[i];
        diff += std::abs(c[i] - cc);
      }
      std::cout << "conditional_assign[scalar,vector] diff = " << diff << "\n";
    }    

    vector_bool_type cond_all_true, cond_all_false, cond_alternate;
    
    for (int i=0;i<vl;++i) {
      cond_all_true[i] = true;
      cond_all_false[i] = false;
      cond_alternate[i] = i%2;
    }
    
    std::cout << "cond true  all true  = " << is_all_true(cond_all_true) << "\n";
    std::cout << "cond true  any true  = " << is_any_true(cond_all_true) << "\n";

    std::cout << "cond false all true  = " << is_all_true(cond_all_false) << "\n";
    std::cout << "cond false any true  = " << is_any_true(cond_all_false) << "\n";

    std::cout << "cond alt   all true  = " << is_all_true(cond_alternate) << "\n";
    std::cout << "cond alt   any true  = " << is_any_true(cond_alternate) << "\n";

    value_type min_a = a[0], max_a = a[0], sum_a = 0, prod_a = 1;
    for (int i=0;i<vl;++i) {
      min_a = min(min_a, a[i]);
      max_a = max(max_a, a[i]);
      sum_a += a[i];
      prod_a *= a[i];
    }

    std::cout << "min(a) = " << min(a) << " " << min_a << "\n";
    std::cout << "max(a) = " << max(a) << " " << max_a << "\n";
    std::cout << "sum(a) = " << sum(a) << " " << sum_a << "\n";
    std::cout << "prod(a) = " << prod(a) << " " << prod_a << "\n";
  }


}

int main(int argc, char *argv[]) {

  Kokkos::initialize();

  Test();

  Kokkos::finalize();

  return 0;
}
