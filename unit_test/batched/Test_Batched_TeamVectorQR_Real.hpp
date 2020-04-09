
#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F( TestCategory, batched_scalar_team_vector_qr_float ) {
  typedef Algo::QR::Unblocked algo_tag_type;
  test_batched_qr<TestExecSpace,float,algo_tag_type>();
}
#endif


#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F( TestCategory, batched_scalar_team_vector_qr_double ) {
  typedef Algo::LU::Unblocked algo_tag_type;
  test_batched_qr<TestExecSpace,double,algo_tag_type>();
}
#endif

