//
// Created by Poliakoff, David Zoeller on 4/26/21.
//

//
// Basic kernels...
//
#include "basic/ATOMIC_PI.hpp"
#include "basic/DAXPY.hpp"
#include "basic/IF_QUAD.hpp"
#include "basic/INIT3.hpp"
#include "basic/INIT_VIEW1D.hpp"
#include "basic/INIT_VIEW1D_OFFSET.hpp"
#include "basic/MULADDSUB.hpp"
#include "basic/NESTED_INIT.hpp"
#include "basic/REDUCE3_INT.hpp"
#include "basic/TRAP_INT.hpp"

//
// Lcals kernels...
//
#include "lcals/DIFF_PREDICT.hpp"
#include "lcals/EOS.hpp"
#include "lcals/FIRST_DIFF.hpp"
#include "lcals/FIRST_MIN.hpp"
#include "lcals/FIRST_SUM.hpp"
#include "lcals/GEN_LIN_RECUR.hpp"
#include "lcals/HYDRO_1D.hpp"
#include "lcals/HYDRO_2D.hpp"
#include "lcals/INT_PREDICT.hpp"
#include "lcals/PLANCKIAN.hpp"
#include "lcals/TRIDIAG_ELIM.hpp"

//
// Polybench kernels...
//
#include "polybench/POLYBENCH_2MM.hpp"
#include "polybench/POLYBENCH_3MM.hpp"
#include "polybench/POLYBENCH_ADI.hpp"
#include "polybench/POLYBENCH_ATAX.hpp"
#include "polybench/POLYBENCH_FDTD_2D.hpp"
#include "polybench/POLYBENCH_FLOYD_WARSHALL.hpp"
#include "polybench/POLYBENCH_GEMM.hpp"
#include "polybench/POLYBENCH_GEMVER.hpp"
#include "polybench/POLYBENCH_GESUMMV.hpp"
#include "polybench/POLYBENCH_HEAT_3D.hpp"
#include "polybench/POLYBENCH_JACOBI_1D.hpp"
#include "polybench/POLYBENCH_JACOBI_2D.hpp"
#include "polybench/POLYBENCH_MVT.hpp"

//
// Stream kernels...
//
#include "stream/COPY.hpp"
#include "stream/MUL.hpp"
#include "stream/ADD.hpp"
#include "stream/TRIAD.hpp"
#include "stream/DOT.hpp"

//
// Apps kernels...
//
#include "apps/WIP-COUPLE.hpp"
#include "apps/DEL_DOT_VEC_2D.hpp"
#include "apps/ENERGY.hpp"
#include "apps/FIR.hpp"
#include "apps/HALOEXCHANGE.hpp"
#include "apps/LTIMES.hpp"
#include "apps/LTIMES_NOVIEW.hpp"
#include "apps/PRESSURE.hpp"
#include "apps/VOL3D.hpp"

//
// Algorithm kernels...
//
#include "algorithm/SORT.hpp"
#include "algorithm/SORTPAIRS.hpp"


#include <iostream>
void make_perfsuite_executor(rajaperf::Executor *exec, int argc, char *argv[]) {
    RunParams run_params(argc, argv);
    free_register_group(exec, std::string("Basic"));
    free_register_group(exec, std::string("Lcals"));
    free_register_group(exec, std::string("Polybench"));
    free_register_group(exec, std::string("Stream"));
    free_register_group(exec, std::string("Apps"));
    free_register_group(exec, std::string("Algorithm"));

    // Basic

    free_register_kernel(exec, "Basic", new basic::ATOMIC_PI(run_params));
    free_register_kernel(exec, "Basic", new basic::DAXPY(run_params));
    free_register_kernel(exec, "Basic", new basic::IF_QUAD(run_params));
    free_register_kernel(exec, "Basic", new basic::INIT3(run_params));
    free_register_kernel(exec, "Basic", new basic::INIT_VIEW1D(run_params));
    free_register_kernel(exec, "Basic", new basic::INIT_VIEW1D_OFFSET(run_params));
    free_register_kernel(exec, "Basic", new basic::MULADDSUB(run_params));
    free_register_kernel(exec, "Basic", new basic::NESTED_INIT(run_params));
    free_register_kernel(exec, "Basic", new basic::REDUCE3_INT(run_params));
    free_register_kernel(exec, "Basic", new basic::TRAP_INT(run_params));
    /**
    // Lcals
    free_register_kernel(exec, "Lcals", new lcals::DIFF_PREDICT(run_params));
    free_register_kernel(exec, "Lcals", new lcals::EOS(run_params));
    free_register_kernel(exec, "Lcals", new lcals::FIRST_DIFF(run_params));
    free_register_kernel(exec, "Lcals", new lcals::FIRST_MIN(run_params));
    free_register_kernel(exec, "Lcals", new lcals::FIRST_SUM(run_params));
    free_register_kernel(exec, "Lcals", new lcals::GEN_LIN_RECUR(run_params));
    free_register_kernel(exec, "Lcals", new lcals::HYDRO_1D(run_params));
    free_register_kernel(exec, "Lcals", new lcals::HYDRO_2D(run_params));
    free_register_kernel(exec, "Lcals", new lcals::INT_PREDICT(run_params));
    free_register_kernel(exec, "Lcals", new lcals::PLANCKIAN(run_params));
    free_register_kernel(exec, "Lcals", new lcals::TRIDIAG_ELIM(run_params));

    // Polybench
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_2MM(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_3MM(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_ADI(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_ATAX(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_FDTD_2D(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_FLOYD_WARSHALL(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_GEMM(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_GEMVER(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_GESUMMV(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_HEAT_3D(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_JACOBI_1D(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_JACOBI_2D(run_params));
    free_register_kernel(exec, "Polybench", new polybench::POLYBENCH_MVT(run_params));

    // Stream
    free_register_kernel(exec, "Stream", new stream::ADD(run_params));
    free_register_kernel(exec, "Stream", new stream::COPY(run_params));
    free_register_kernel(exec, "Stream", new stream::DOT(run_params));
    free_register_kernel(exec, "Stream", new stream::MUL(run_params));
    free_register_kernel(exec, "Stream", new stream::TRIAD(run_params));

    // Apps
    free_register_kernel(exec, "Apps", new apps::COUPLE(run_params));
    free_register_kernel(exec, "Apps", new apps::DEL_DOT_VEC_2D(run_params));
    free_register_kernel(exec, "Apps", new apps::ENERGY(run_params));
    free_register_kernel(exec, "Apps", new apps::FIR(run_params));
    free_register_kernel(exec, "Apps", new apps::HALOEXCHANGE(run_params));
    free_register_kernel(exec, "Apps", new apps::LTIMES(run_params));
    free_register_kernel(exec, "Apps", new apps::LTIMES_NOVIEW(run_params));
    free_register_kernel(exec, "Apps", new apps::PRESSURE(run_params));
    free_register_kernel(exec, "Apps", new apps::VOL3D(run_params));

    // Algorithm
    free_register_kernel(exec, "Algorithm", new algorithm::SORT(run_params));
    free_register_kernel(exec, "Algorithm", new algorithm::SORTPAIRS(run_params));
    */
}

