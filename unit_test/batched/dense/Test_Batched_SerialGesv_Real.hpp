#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_serial_gesv_float) {
  test_batched_gesv<TestExecSpace, float>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_serial_gesv_double) {
  test_batched_gesv<TestExecSpace, double>();
}
#endif