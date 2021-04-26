//
// Created by Poliakoff, David Zoeller on 4/26/21.
//
#include <common/RAJAPerfSuite.hpp>
#include <common/Executor.hpp>
#include "sparse/tracked_testing.hpp"
#include <iostream>

int main(int argc, char* argv[]){
  rajaperf::Executor exec(argc, argv);
  rajaperf::RunParams run_params(argc, argv);
  test::sparse::build_executor(exec, argc, argv, run_params);
  exec.setupSuite();

  // STEP 3: Report suite run summary
  //         (enable users to catch errors before entire suite is run)
  exec.reportRunSummary(std::cout);

  // STEP 4: Execute suite
  exec.runSuite();

  // STEP 5: Generate suite execution reports
  exec.outputRunData();

  std::cout << "\n\nDONE!!!...." << std::endl;

}