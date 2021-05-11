//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_Executor_HPP
#define RAJAPerf_Executor_HPP

#include "common/RAJAPerfSuite.hpp"
#include "common/RunParams.hpp"

#include <map>
#include <iosfwd>
#include <utility>
#include <set>

namespace rajaperf {

class KernelBase;
class WarmupKernel;

/*!
 *******************************************************************************
 *
 * \brief Class that assembles kernels and variants to run and executes them.
 *
 *******************************************************************************
 */
class Executor
{
public:
  Executor( int argc, char** argv );

  ~Executor();

  void setupSuite();

  void reportRunSummary(std::ostream& str) const;

  void runSuite();
  
  void outputRunData();

  // Interface for adding new Kokkos groups and kernels 

  using groupID = int;
  using kernelSet = std::set<KernelBase*>;
  using kernelMap = std::map<std::string, KernelBase*>;
  using groupMap =  std::map<std::string, kernelSet>;
  using kernelID = int;

  ///////////////////////////////////////////////////
  //
  // Logic:
  // Need the full set of kernels
  // Associate group names (e.g., lcals, basic) with kernel sets
  // Interface to add new kernels (e.g., DAXPY) and groups (basic) 
  // for Kokkos Performance Testing 

  groupID registerGroup(std::string groupName);

  kernelID registerKernel(std::string, KernelBase*);

  std::vector<KernelBase*> lookUpKernelByName(std::string kernelOrGroupName);

private:
  Executor() = delete;

  enum CSVRepMode {
    Timing = 0,
    Speedup,

    NumRepModes // Keep this one last and DO NOT remove (!!)
  };

  struct FOMGroup {
    VariantID base;
    std::vector<VariantID> variants;
  }; 

  bool haveReferenceVariant() { return reference_vid < NumVariants; }

  void writeCSVReport(const std::string& filename, CSVRepMode mode, 
                      size_t prec);
  std::string getReportTitle(CSVRepMode mode);
  long double getReportDataEntry(CSVRepMode mode, 
                                 KernelBase* kern, VariantID vid);

  void writeChecksumReport(const std::string& filename);  

  void writeFOMReport(const std::string& filename);
  void getFOMGroups(std::vector<FOMGroup>& fom_groups);
  
 // Kokkos add group and kernel ID inline functions
 // Provisional Design for Kokkos
 
  inline groupID getNewGroupID() {
          // The newGroupID will be shared amongst invocations of this
          // function.
        static groupID newGroupID;

        return newGroupID++;

  }

  inline kernelID getNewKernelID() {
        
        static kernelID newKernelID;
        return newKernelID++;

  }



  // Data members
  RunParams run_params;
  std::vector<KernelBase*> kernels;  
  std::vector<VariantID>   variant_ids;

  VariantID reference_vid;

  // "allKernels" is an instance of kernelMap, which is a "map" of all kernels (as strings, e.g., DAXPY, to their
  // kernelBase* instances; the string name will be the key (first), and the kernelBase* instance will be the value (second)
  kernelMap allKernels;
  // "kernelsPerGroup" is an instance of "groupMap;" "kernelsPerGroup" maps kernels to their
  // categories / parent class (e.g., basic, polybench, etc.)
  groupMap kernelsPerGroup;


};

void free_register_group(Executor*, std::string);
void free_register_kernel(Executor*, std::string, KernelBase*);

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
