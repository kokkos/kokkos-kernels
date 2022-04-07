#if defined(KOKKOSKERNELS_INST_FLOAT)
TEST_F(TestCategory, batched_scalar_teamvector_gesv_float) {
  test_batched_teamvector_gesv<TestExecSpace, float>();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE)
TEST_F(TestCategory, batched_scalar_teamvector_gesv_double) {
  test_batched_teamvector_gesv<TestExecSpace, double>();
}
#endif