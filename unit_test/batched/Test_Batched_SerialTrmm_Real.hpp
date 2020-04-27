
#if defined(KOKKOSKERNELS_INST_FLOAT)
// NO TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_nt_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_nt_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_nt_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_nt_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_nt_u_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_nt_n_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
// TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_t_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_t_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_t_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_t_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_t_u_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_t_n_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
// CONJUGATE TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_ct_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_ct_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_ct_u_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_ct_n_float_float ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_ct_u_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_ct_n_float_float ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,float,float,param_tag_type,algo_tag_type,param_trans_type>();
}
#endif


#if defined(KOKKOSKERNELS_INST_DOUBLE)
// NO TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_nt_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_nt_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_nt_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_nt_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_nt_u_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_nt_n_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::NoTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
// TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_t_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_t_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_t_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_t_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_t_u_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::Transpose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_t_n_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::Transpose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::Transpose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
// CONJUGATE TRANSPOSE
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_ct_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_l_ct_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Lower,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_ct_u_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_l_u_ct_n_double_double ) {
  typedef ::Test::ParamTag<Side::Left,Uplo::Upper,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_ct_u_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::ConjTranspose,Diag::Unit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
TEST_F( TestCategory, batched_scalar_serial_trmm_r_u_ct_n_double_double ) {
  typedef ::Test::ParamTag<Side::Right,Uplo::Upper,Trans::ConjTranspose,Diag::NonUnit> param_tag_type;
  typedef Algo::Trsm::Blocked algo_tag_type;
  typedef Trans::ConjTranspose param_trans_type;
  test_batched_trmm<TestExecSpace,double,double,param_tag_type,algo_tag_type,param_trans_type>();
}
#endif

