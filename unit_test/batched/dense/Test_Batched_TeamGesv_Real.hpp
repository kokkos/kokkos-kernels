#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_team_gesv_float) {
  test_batched_team_gesv<TestExecSpace, float>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_team_gesv_double) {
  test_batched_team_gesv<TestExecSpace, double>();
}
#endif