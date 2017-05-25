#!/bin/bash
KOKKOSKERNELS_PATH=$1
cd ${KOKKOSKERNELS_PATH}/src/impl
mkdir generated_specializations_hpp
mkdir generated_specializations_cpp

#abs
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash abs KokkosBlas1_abs KokkosBlas1_abs_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash abs KokkosBlas1_abs_mv KokkosBlas1_abs_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#axpby
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash axpby KokkosBlas1_axpby KokkosBlas1_axpby_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash axpby KokkosBlas1_axpby_mv KokkosBlas1_axpby_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#dot
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash dot KokkosBlas1_dot KokkosBlas1_dot_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash dot KokkosBlas1_dot_mv KokkosBlas1_dot_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#nrm2
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2 KokkosBlas1_nrm2 KokkosBlas1_nrm2_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2 KokkosBlas1_nrm2_mv KokkosBlas1_nrm2_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#scal
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash scal KokkosBlas1_scal KokkosBlas1_scal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash scal KokkosBlas1_scal_mv KokkosBlas1_scal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
