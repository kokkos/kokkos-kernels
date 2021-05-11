//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "Executor.hpp"

#include "common/KernelBase.hpp"
#include "common/OutputUtils.hpp"

// Warmup kernel to run first to remove startup overheads in timings
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
#include "basic/DAXPY.hpp"
#endif

// Standard library includes
#include <list>
#include <vector>
#include <string>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>

#include <unistd.h>


namespace rajaperf {

using namespace std;

Executor::Executor(int argc, char** argv)
  : run_params(argc, argv),
    reference_vid(NumVariants)
{
}

/*
 * https://www.delftstack.com/howto/cpp/cpp-tilde-operator/
 *
 * The destructor is a special member function that handles the deallocation of the class object’s resources.
 * AS opposed to the class constructors, it has only one destructor function for a given class.
 * The class destructor is declared with the same name as the class plus the prefix ~ tilde operator.
 *...
 * Generally, the class members are destroyed after the destructor function code is run;
 * thus, we can demonstrate how the StringArray class instance goes out of scope and hence
 * printing to the console the corresponding text.
 *
 */

// Destructor for resource de-allocation

Executor::~Executor()
{
  for (size_t ik = 0; ik < kernels.size(); ++ik) {
    delete kernels[ik];
  }

  // Pre-processor directives
#if defined(RUN_KOKKOS)
  Kokkos::finalize(); // TODO DZP: should this be here?  Good question.  AJP
#endif
}

// New functions for Kokkos to register new group and kernel IDs
// The return type is Executor::groupID


Executor::groupID Executor::registerGroup(std::string groupName)
{
   // find() method searches the string for the first occurrence of the sequence specified by its arguments.
   // Recall, "kernelsPerGroup" is a mapping of kernel groups (e.g., basic) and their constituent kernels (e.g., DAXPY)
   auto checkIfGroupExists = kernelsPerGroup.find(groupName);


    /* Recall, these items are defined in Executor.hpp:
    using groupID = int;
    using kernelID = int;
    using kernelSet = std::set<KernelBase*>;                // data type: set of KernelBase* instances
    using kernelMap = std::map<std::string, KernelBase*>;   // data type:  map of string kernel names to instances of KernelBase*
    using groupMap =  std::map<std::string, kernelSet>;     // data type: map of groupNames to sets of kernels
     ...
     // "allKernels" is an instance of kernelMap, which is a "map" of all kernels and their ID's
     kernelMap allKernels;

     // "kernelsPerGroup" is an instance of "groupMap;" "kernelsPerGroup" maps kernels to their categories (e.g., basic, polybench, etc.)
     groupMap kernelsPerGroup;

    */

    /*  end()
 * Return iterator to end
 * Returns an iterator referring to the past-the-end element in the vector container.
 * The past-the-end element is the theoretical element that would follow the last element in the vector.
 * It does not point to any element, and thus shall not be de-referenced.
 * Because the ranges used by functions of the standard library do not include
 * the element pointed by their closing iterator,
 * this function is often used in combination with vector::begin to specify a range including all the elements in the container.
 * If the container is empty, this function returns the same as vector::begin.
 *
 */


    // HERE, WE ARE CHECKING THE CASE THAT THE groupNAME **IS NOT** IN THE MAP OBJECT
    // Using the .end() idiom to check if I've fallen off the edge of the container without finding a match
    if (checkIfGroupExists == kernelsPerGroup.end()){
        // If groupName not found, set that groupName in kernelsPerGroup to an empty kernelSet obj
      kernelsPerGroup[groupName] = kernelSet();
}
 else {
	// ERROR CONDITION:  DUPLICATING GROUPS
	// Error lists exsiting group, and kills program.

	std::cout << "The Group Name " <<  groupName << " already exists.  Program is exiting." << std::endl;

	// In kernelsPerGroup, the Group Name is the first position / key value, and the second position / value type in the set
   auto fullKernelSet = checkIfGroupExists->second;

   // fullKernelSet is of type std::set<kernelBase*>

   for (auto kernel: fullKernelSet) {

       std::cout << kernel->getName() << std::endl;

   }

	exit(1);
	
}
   // getNewGroupID() is an object of type Executor::groupID, an int
   return getNewGroupID();


}

// New function with return type Executor::kernelID, returning getNewKernelID(); registerKernel is a new function in the Executor class
//

Executor::kernelID Executor::registerKernel(std::string groupName, KernelBase* kernel)
{
  // declaring and setting kernelName to de-referenced kernel pointer obj, an instance of KernelBase*
  auto kernelName = kernel->getName();
  // Recall, "allKernels" maps named kernels to their IDs
  auto checkIfKernelExists = allKernels.find(kernelName);
  // Check if checkKernelExists value IS NOT in  the map of all kernels
  if (checkIfKernelExists == allKernels.end()) {
      // if the kernel name IS NOT in the allKernels map, set kernelName to kernel, the KernelBase* instance
     allKernels[kernelName] = kernel;
} 
  else {
      // ERROR CONDITION:  if the kernel is found / exists, make the program exit

      std::cout << "Kernel " << checkIfKernelExists->first << " already exists.  Program is exiting." << std::endl;

      exit(1);
  }
  //////////////////////////////////////////////////////////////////////////////
   // This error condition : adding a groupName before checking if the group associated with the kernel exists
   // Declare and set checkIfGroupExists to the value of the string-type groupName in the kernelsPerGroup map
   auto checkIfGroupExists = kernelsPerGroup.find(groupName);
   // LOGIC:  Check if checkIfGroupExists value is the same as the past-the-end element in the vector container, which
   // does not have a value
   // i.e., check for the case that the groupName DOES NOT exist with the ".end()" idiom;
   if (checkIfGroupExists == kernelsPerGroup.end()){

}

else {
  // If the groupName DOES EXIST, then insert the kernel (instance of KernelBase*) at the second position of the
  // allKernels map to associate the kernel and its groupNAme

  checkIfGroupExists -> second.insert(kernel);

}

    // getNewKernelID is an obj of type Executor::kernelID
    return getNewKernelID();
}

// AJP & DZP new function
// AJP GOAL:  return a vector of all kernelBase* objects to be run by <WHICH METHODS??>

std::vector<KernelBase*>  Executor::lookUpKernelByName(std::string kernelOrGroupName){

	// The vector / list return type, std::vector<KernelBase*>  will contain 
	// either all of the kernels with  a given  kernel name or group name
	// We have two maps (defined in Executor.hpp): kernelMap allKernels, groupMap kernelsPerGroup, 
	// STEPS:
	// 1) declare new vector that will contain the string data:
	// 2) LOGIC:
	// 	i) check to see if the kernel / group requested on the 
	// 	"./rajaperf.exe -k" line (you can pass either a specific kernel or a
	// 	kernel groupName, e.g., "Basic"

	// Declaring the vector kernelsByNameVect of type std::vector<KernelBase*>;
	// This variable will contain the set of kernels to run
   std::vector<KernelBase*> kernelsByNameVect ;

        // CONDITIONS TO INCLUDE:
        // 1)  If kernelName is groupName , then add that set of kernels in the
        // group to the vector

        // 2) else if kernelName is kernel, then add the kernel to the vector
        // 3) else if kernelName is horse stuff, then say so

        // HINT:  Declare iterator against which you can test equivalence

        auto checkLookUpGroupNameIterator =  kernelsPerGroup.find(kernelOrGroupName);
        auto checkLookUpKernelNameIterator = allKernels.find(kernelOrGroupName);

        // Check to see if groupName NOT in kernelsPerGroup;
        // end() iterates to the end 
        if (checkLookUpGroupNameIterator != kernelsPerGroup.end()) {
                //cout << " STEP 1" << endl;
                
                // when using the arrow, you get a key, value pair.
                // You can access either member by "first" or "second"

                // we have std::set of KernelBase*
                auto groupSetForTests = checkLookUpGroupNameIterator -> second;

                for (auto item: groupSetForTests) {
                        kernelsByNameVect.push_back(item);
                        }
        }

        else if (checkLookUpKernelNameIterator != allKernels.end()) {

                auto kernel = checkLookUpKernelNameIterator -> second;

                        kernelsByNameVect.push_back(kernel);


        }


	// kernelsByNameVect is an object of type std::vector<KernelBase*> that will be used by <void Executor::setupSuite()>
    return kernelsByNameVect;
	
 
}



//////////////////////////////////////////////////////////////////////////////////////
// * AJP TASK:  change the setupSuite to use the allKernels (type:  kernelMap) and kernelsPerGroup (type: groupMap)
// * maps;
// * The goal here is to make a vector of the different instances of KernelBase*, kernel, that are to be run;
// * The vector you'll need already exists!
// * Hint: see line 375-ish for kernels.push_back;
// */
/////////////////////////////////////////////////////////////////////////////////////
void Executor::setupSuite()
{
    // Initial handling of run parameters input
  RunParams::InputOpt in_state = run_params.getInputState();
  // QUESTION -- In this first step, are we doing nothing (initially) if we have bad input?
  // Should there be an else condition for this conditional?
  if ( in_state == RunParams::InfoRequest || in_state == RunParams::BadInput ) {
    return;
  }

  cout << "\nSetting up suite based on input..." << endl;


  ////////////////////////////////////////////////////////////////////////////////////
  // Declaring function type aliases

  using Slist = list<string>;
  using Svector = vector<string>;
  // Set of kernel IDs, e.g., DAXPY, IF_QUAD
  // "variants" include CUDA, OpenMPTarget, OpenMP, HIP, Serial
  using VIDset = set<VariantID>;
  ///////////////////////////////////////////////////////////////////////////////////
  // Determine which kernels to execute from input.
  // run_kern will be non-duplicated ordered set of IDs of kernel to run.
  // kernel_input is an object of type reference to Svector;
  // kernel_input will contain the input for the kernels to run
  const Svector& kernel_input = run_params.getKernelInput();

  // Declare run_kern of type KIDset; contains the set of kernels (KernelBase* instances to run)

  /* LOGIC
  1) check if each of the inputs in matches a groupName;
  2) if a match, add every kernel in that group to the vector that will be run;
  3) if no match, check existing kernels
  4) if a match, add that kernel
  5) if no match, add that kernel to set the set of invalid kernels
  */

    Svector invalid;

    // The case when the executable is passed no args
    if (kernel_input.empty()) {
        // your iterator does the deferencing for you, thus you don't need the input arrow, which is
        // necessary for dereferencing

        for (auto iter_input: allKernels) {
            kernels.push_back(iter_input.second);
        }
    }
    else {

        for (auto kernelName: kernel_input) {
            std::vector<KernelBase *> matchingKernelsVec = lookUpKernelByName(kernelName);
            // if everything that matched is in the vector, and nothing matched, i.e., an empty vector,
            // i.e., the kernel name was invalid

            if (matchingKernelsVec.empty()) {
                invalid.push_back(kernelName);
            } else {

                for (auto iter_kern: matchingKernelsVec) {
                    kernels.push_back(iter_kern);

                }
            }
        }
    }

/*
  if ( kernel_input.empty() ) {

    //
    // if No kernels specified in input, run them all...
    //
    for (size_t ik = 0; ik < NumKernels; ++ik) {
        // here, inserting kernels to run; you must cast ik (of type size_t), the indexing variable, as a KernelID type
      run_kern.insert( static_cast<KernelID>(ik) );
    }

  } else {


    // Parse input to determine which kernels to run
    // Make list of strings copy of kernel input for the parsing
    // (need to process potential group names and/or kernel names)

    // Slist is a type alias for list<string>
    // Populate list with the kernel_input, from the beginning index to the end
    Slist input(kernel_input.begin(), kernel_input.end());

    // AJP code addition -- print list of inputs

    for (auto idx: input )

        std::cout << "Input parameters list:  " << idx << std:: endl;

    // Search input for matching group names.
    // groups2run is a vector of strings (of type Svector, a type alias of vector<strings>) containing names
    // of groups to run if passed in as input.

    Svector groups2run;
    // Outer loop:  Iterate through the list of strings from the first to the last item
    for (Slist::iterator it = input.begin(); it != input.end(); ++it) {
      // inner loop:  iterate over NumGroups, a member of GroupID enum defined in RAJAPerfSuite.hpp
      for (size_t ig = 0; ig < NumGroups; ++ig) {
          // declare a constant (immutable) string reference "group_name"
          // Store the value at the the ig(th) index as a GroupID in group_name
        const string& group_name = getGroupName(static_cast<GroupID>(ig));
        // if group_name is equal to the value the it(th)* index points to,
        // push_back / append that group_name to groups2run vector of strings
        if ( group_name == *it ) {
          groups2run.push_back(group_name);
        }
      }
    }

    // If group name(s) found in input, assemble kernel sets for those group(s);
    // to run and remove those group name(s) from input list.
    // Here, iterate the groups2run, and store the value at ig(th) index in
    // an immutable/constant reference called gname (of type string)
    for (size_t ig = 0; ig < groups2run.size(); ++ig) {
      const string& gname(groups2run[ig]);

      // NumKernels is always the last member of KernelID, an enum, declared in RAJAPerfSuite.hpp
      // Iterate over NumKernels, casting the index ik to a KernelID type, and setting it to kid
      //
      for (size_t ik = 0; ik < NumKernels; ++ik) {
        KernelID kid = static_cast<KernelID>(ik);
        // if the group name DOES occur within the string full kernel name (npos means until the end of the string),
        // insert the kid (of KernelID type) into the run_kern (of type KIDset)
        if ( getFullKernelName(kid).find(gname) != string::npos ) {
          run_kern.insert(kid);
        }
      }
      // remember, gname is a const/immutable string reference containing group names as a string
      input.remove(gname);
    }



    // Look for matching names of individual kernels in remaining input.
    // Assemble invalid input for warning message.
    // Declare the vector "invalid" of type Svector (type alias for vector<string>) to hold ...
    // Iterate over the input from beginning to the end item;
    for (Slist::iterator it = input.begin(); it != input.end(); ++it) {
        // initialize a boolean, "found_it" to false;
        // why do we need this variable? AJP -- ANSWER HERE
      bool found_it = false;
      // Iterate ik over NumKernels & TRUE;
      // Iterate until you hit the end of the list , or until you find what you're looking for.
      for (size_t ik = 0; ik < NumKernels && !found_it; ++ik) {
          // cast the ik(th) value to a KernelID, and set equal to kid
        KernelID kid = static_cast<KernelID>(ik);
        // if the kernel name (for a kid, of type KernelID) is equal to the value pointed at at the it(th) index
        // OR if the full kernel name (for a kid) is equal to the value pointed at at the it(th) index
        // insert that kid into the run_kern (of type KIDset) and set found_it boolean to true
        if ( getKernelName(kid) == *it || getFullKernelName(kid) == *it ) {
          run_kern.insert(kid);
          found_it = true;
        }
      }
      // ATTN: found_it depend on whether or not the kernel was found;
      // if the kernel was NOT found, we want to push it back to the set of invalid;
      // if found_it = false, push back the value pointed at at the it(th) index to the vector of strings, "&invalid,"
      // which is of type Svector (a type alias)
      if ( !found_it )  invalid.push_back(*it); 
    }
    //  Update the run_params obj with data in the invalid vector reference
    run_params.setInvalidKernelInput(invalid);

  }

  //
  // Assemble set of available variants to run 
  // (based on compile-time configuration).
  // Recall, a variant will be:  base_seq, base_CUDA, Raja_lambda, kokkos_lambda, etc.

  // Declare available_var as a VIDset
*/

  run_params.setInvalidKernelInput(invalid);

  VIDset available_var;
  // iterate the NumVariants & static_cast value at iv(th) index to VariantID
  // if the variant is available, insert vid into the VIDset
  for (size_t iv = 0; iv < NumVariants; ++iv) {
    VariantID vid = static_cast<VariantID>(iv);
    if ( isVariantAvailable( vid ) ) {
       available_var.insert( vid );
    }
  }

  //
  // Determine variants to execute from input.
  // run_var will be non-duplicated ordered set of IDs of variants to run.
  //
  const Svector& variant_input = run_params.getVariantInput();

  VIDset run_var;

  if ( variant_input.empty() ) {

    //
    // No variants specified in input options, run all available.
    // Also, set reference variant if specified.
    //
    for (VIDset::iterator vid_it = available_var.begin();
         vid_it != available_var.end(); ++vid_it) {
      VariantID vid = *vid_it;
      run_var.insert( vid );
      if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
        reference_vid = vid;
      }
    }

    //
    // Set reference variant if not specified.
    // Here, this is where base_seq is set as the default baseline;
    // the baseline that is used can be changed!
    // e.g., kokkos_lambda

    if ( run_params.getReferenceVariant().empty() && !run_var.empty() ) {
      reference_vid = *run_var.begin();
    }

  } else {


    //
    // Parse input to determine which variants to run:
    //   - variants to run will be the intersection of available variants
    //     and those specified in input
    //   - reference variant will be set to specified input if available
    //     and variant will be run; else first variant that will be run.
    // 
    // Assemble invalid input for warning message.
    //

    Svector invalid;

    for (size_t it = 0; it < variant_input.size(); ++it) {
      bool found_it = false;

      for (VIDset::iterator vid_it = available_var.begin();
         vid_it != available_var.end(); ++vid_it) {
        VariantID vid = *vid_it;
        if ( getVariantName(vid) == variant_input[it] ) {
          run_var.insert(vid);
          if ( getVariantName(vid) == run_params.getReferenceVariant() ) {
            reference_vid = vid;
          }
          found_it = true;
        }
      }

      if ( !found_it )  invalid.push_back(variant_input[it]);
    }

    //
    // Set reference variant if not specified.
    //
    if ( run_params.getReferenceVariant().empty() && !run_var.empty() ) {
      reference_vid = *run_var.begin();
    }

    run_params.setInvalidVariantInput(invalid);

  }



  if ( !(run_params.getInvalidKernelInput().empty()) ) {

    run_params.setInputState(RunParams::BadInput); 

  } else { // kernel input looks good

      // Get lists using David and Amy's new maps!

/*    for (KIDset::iterator kid = run_kern.begin();
         kid != run_kern.end(); ++kid) {
/// RDH DISABLE COUPLE KERNEL until we find a reasonable way to do 
/// complex numbers in GPU code
      if ( true ) {
        kernels.push_back( getKernelObject(*kid, run_params) );
      }
    }
*/
    if ( !(run_params.getInvalidVariantInput().empty()) ) {

       run_params.setInputState(RunParams::BadInput);

    } else { // variant input lools good

      for (VIDset::iterator vid = run_var.begin();
           vid != run_var.end(); ++vid) {
        variant_ids.push_back( *vid );
      }

      //
      // If we've gotten to this point, we have good input to run.
      //
      if ( run_params.getInputState() != RunParams::DryRun && 
           run_params.getInputState() != RunParams::CheckRun ) {
        run_params.setInputState(RunParams::PerfRun);
      }

    } // kernel and variant input both look good
#if defined(RUN_KOKKOS)
    Kokkos::initialize();
    /** 
     * DZP: This is a terrible hack to just get the push/pop region
     * callbacks without the begin_parallel_x/end_parallel_x ones,
     * so we don't overfence and perturb performance
     */
    auto events = Kokkos::Tools::Experimental::get_callbacks();
    auto push = events.push_region;
    auto pop = events.pop_region;
    auto metadata = events.declare_metadata;
    Kokkos::Tools::Experimental::pause_tools();
    Kokkos::Tools::Experimental::set_push_region_callback(push);
    Kokkos::Tools::Experimental::set_pop_region_callback(pop);
    Kokkos::Tools::Experimental::set_declare_metadata_callback(metadata);
#endif
  } // if kernel input looks good

}
////////////////////////////////////////////////////////////////////////////////////

void Executor::reportRunSummary(ostream& str) const
{
  RunParams::InputOpt in_state = run_params.getInputState();

  if ( in_state == RunParams::BadInput ) {

    str << "\nRunParams state:\n";
    str <<   "----------------";
    run_params.print(str);

    str << "\n\nSuite will not be run now due to bad input."
        << "\n  See run parameters or option messages above.\n" 
        << endl;

  } else if ( in_state == RunParams::PerfRun || 
              in_state == RunParams::DryRun || 
              in_state == RunParams::CheckRun ) {

    if ( in_state == RunParams::DryRun ) {

      str << "\n\nRAJA performance suite dry run summary...."
          <<   "\n--------------------------------------" << endl;
 
      str << "\nInput state:";
      str << "\n------------";
      run_params.print(str);

    } 

    if ( in_state == RunParams::PerfRun ||
         in_state == RunParams::CheckRun ) {

      str << "\n\nRAJA performance suite run summary...."
          <<   "\n--------------------------------------" << endl;

    }

    string ofiles;
    if ( !run_params.getOutputDirName().empty() ) {
      ofiles = run_params.getOutputDirName();
    } else {
      ofiles = string(".");
    }
    ofiles += string("/") + run_params.getOutputFilePrefix() + 
              string("*");

    str << "\nHow suite will be run:" << endl;
    str << "\t # passes = " << run_params.getNumPasses() << endl;
    str << "\t Kernel size factor = " << run_params.getSizeFactor() << endl;
    str << "\t Kernel rep factor = " << run_params.getRepFactor() << endl;
    str << "\t Output files will be named " << ofiles << endl;

#if defined(RUN_KOKKOS)
    Kokkos::Tools::declareMetadata("replication_factor",std::to_string(run_params.getRepFactor()));
    Kokkos::Tools::declareMetadata("size_factor",std::to_string(run_params.getSizeFactor()));
#endif

    str << "\nThe following kernels and variants (when available) will be run:\n"; 

    str << "\nVariants"
        << "\n--------\n";
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      str << getVariantName(variant_ids[iv]) << endl;
    }

    str << "\nKernels(iterations/rep , reps)"
        << "\n-----------------------------\n";
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];
      str << kern->getName() 
          << " (" << kern->getItsPerRep() << " , "
          << kern->getRunReps() << ")" << endl;
    }

  }

  str.flush();
}

void Executor::runSuite()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::PerfRun && 
       in_state != RunParams::CheckRun ) {
    return;
  }

  cout << "\n\nRun warmup kernel...\n";
#ifndef RAJAPERF_INFRASTRUCTURE_ONLY
  KernelBase* warmup_kernel = new basic::DAXPY(run_params);

  for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
    VariantID vid = variant_ids[iv];
    if ( run_params.showProgress() ) {
      if ( warmup_kernel->hasVariantToRun(vid) ) {
        cout << "   Running ";
      } else {
        cout << "   No ";
      }
      cout << getVariantName(vid) << " variant" << endl;
    }
    if ( warmup_kernel->hasVariantToRun(vid) ) {
      warmup_kernel->execute(vid);
    }
  }

  delete warmup_kernel;
#endif

  cout << "\n\nRunning specified kernels and variants...\n";

  const int npasses = run_params.getNumPasses();
  for (int ip = 0; ip < npasses; ++ip) {
    if ( run_params.showProgress() ) {
      std::cout << "\nPass through suite # " << ip << "\n";
    }
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kernel = kernels[ik];
      if ( run_params.showProgress() ) {
        std::cout << "\nRun kernel -- " << kernel->getName() << "\n"; 
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
         VariantID vid = variant_ids[iv];
         KernelBase* kern = kernels[ik];
         if ( run_params.showProgress() ) {
           if ( kern->hasVariantToRun(vid) ) {
             cout << "   Running ";
           } else {
             cout << "   No ";
           }
           cout << getVariantName(vid) << " variant" << endl;
         }
         if ( kern->hasVariantToRun(vid) ) {
           kernels[ik]->execute(vid);
         }
      } // loop over variants 

    } // loop over kernels

  } // loop over passes through suite

}

void Executor::outputRunData()
{
  RunParams::InputOpt in_state = run_params.getInputState();
  if ( in_state != RunParams::PerfRun && 
       in_state != RunParams::CheckRun ) {
    return;
  }

  cout << "\n\nGenerate run report files...\n";

  //
  // Generate output file prefix (including directory path). 
  //
  string out_fprefix;
  string outdir = recursiveMkdir(run_params.getOutputDirName()); 
  if ( !outdir.empty() ) {
    chdir(outdir.c_str());
  }
  out_fprefix = "./" + run_params.getOutputFilePrefix();

  string filename = out_fprefix + "-timing.csv";
  writeCSVReport(filename, CSVRepMode::Timing, 6 /* prec */);

  if ( haveReferenceVariant() ) { 
    filename = out_fprefix + "-speedup.csv";
    writeCSVReport(filename, CSVRepMode::Speedup, 3 /* prec */);
  }

  filename = out_fprefix + "-checksum.txt";
  writeChecksumReport(filename);

  filename = out_fprefix + "-fom.csv";
  writeFOMReport(filename);
}


void Executor::writeCSVReport(const string& filename, CSVRepMode mode,
                              size_t prec)
{
  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string kernel_col_name("Kernel  ");
    const string sepchr(" , ");

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()); 
    }
    kercol_width++;

    vector<size_t> varcol_width(variant_ids.size());
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      varcol_width[iv] = max(prec+2, getVariantName(variant_ids[iv]).size()); 
    } 

    //
    // Print title line.
    //
    file << getReportTitle(mode);

    //
    // Wrtie CSV file contents for report.
    // 

    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      file << sepchr;
    }
    file << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      file << sepchr <<left<< setw(varcol_width[iv])
           << getVariantName(variant_ids[iv]);
    }
    file << endl;

    //
    // Print row of data for variants of each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];
      file <<left<< setw(kercol_width) << kern->getName();
      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
        file << sepchr <<right<< setw(varcol_width[iv]);
        if ( (mode == CSVRepMode::Speedup) &&
             (!kern->hasVariantToRun(reference_vid) || 
              !kern->hasVariantToRun(vid)) ) {
          file << "Not run";
        } else if ( (mode == CSVRepMode::Timing) && 
                    !kern->hasVariantToRun(vid) ) {
          file << "Not run";
        } else {
          file << setprecision(prec) << std::fixed 
               << getReportDataEntry(mode, kern, vid);
        }
      }
      file << endl;
    }

    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeFOMReport(const string& filename)
{
  vector<FOMGroup> fom_groups; 
  getFOMGroups(fom_groups);
  if (fom_groups.empty() ) {
    return;
  }

  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string kernel_col_name("Kernel  ");
    const string sepchr(" , ");
    size_t prec = 2;

    size_t kercol_width = kernel_col_name.size();
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      kercol_width = max(kercol_width, kernels[ik]->getName().size()); 
    }
    kercol_width++;

    size_t fom_col_width = prec+14;

    size_t ncols = 0;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      ncols += group.variants.size(); // num variants to compare 
                                      // to each PM baseline
    }

    vector<int> col_exec_count(ncols, 0);
    vector<double> col_min(ncols, numeric_limits<double>::max());
    vector<double> col_max(ncols, -numeric_limits<double>::max());
    vector<double> col_avg(ncols, 0.0);
    vector<double> col_stddev(ncols, 0.0);
    vector< vector<double> > pct_diff(kernels.size());
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      pct_diff[ik] = vector<double>(ncols, 0.0);
    }

    //
    // Print title line.
    //
    file << "FOM Report : signed speedup(-)/slowdown(+) for each PM (base vs. RAJA) -> (T_RAJA - T_base) / T_base )";
    for (size_t iv = 0; iv < ncols*2; ++iv) {
      file << sepchr;
    }
    file << endl;

    file << "'OVER_TOL' in column to right if RAJA speedup is over tolerance";
    for (size_t iv = 0; iv < ncols*2; ++iv) {
      file << sepchr;
    }
    file << endl;

    string pass(",        ");
    string fail(",OVER_TOL");

    //
    // Print column title line.
    //
    file <<left<< setw(kercol_width) << kernel_col_name;
    for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
      const FOMGroup& group = fom_groups[ifg];
      for (size_t gv = 0; gv < group.variants.size(); ++gv) {
        string name = getVariantName(group.variants[gv]);
        file << sepchr <<left<< setw(fom_col_width) << name << pass; 
      } 
    }
    file << endl;


    //
    // Write CSV file contents for FOM report.
    // 

    //
    // Print row of FOM data for each kernel.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];          

      file <<left<< setw(kercol_width) << kern->getName();
     
      int col = 0;
      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        VariantID base_vid = group.base;

        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID comp_vid = group.variants[gv];

          //
          // If kernel variant was run, generate data for it and
          // print (signed) percentage difference from baseline.
          // 
          if ( kern->wasVariantRun(comp_vid) ) {
            col_exec_count[col]++;

            pct_diff[ik][col] = 
              (kern->getTotTime(comp_vid) - kern->getTotTime(base_vid)) /
               kern->getTotTime(base_vid);

            string pfstring(pass);
            if (pct_diff[ik][col] > run_params.getPFTolerance()) {
              pfstring = fail;
            }

            file << sepchr << setw(fom_col_width) << setprecision(prec)
                 <<left<< pct_diff[ik][col] <<right<< pfstring;

            //
            // Gather data for column summaries (unsigned).
            //  
            col_min[col] = min( col_min[col], pct_diff[ik][col] );
            col_max[col] = max( col_max[col], pct_diff[ik][col] );
            col_avg[col] += pct_diff[ik][col];

          } else {  // variant was not run, print a big fat goose egg...

            file << sepchr <<left<< setw(fom_col_width) << setprecision(prec)
                 << 0.0 << pass;

          }

          col++;

        }  // loop over group variants

      }  // loop over fom_groups (i.e., columns)

      file << endl;

    } // loop over kernels


    // 
    // Compute column summary data.
    // 

    // Column average...
    for (size_t col = 0; col < ncols; ++col) {
      if ( col_exec_count[col] > 0 ) {
        col_avg[col] /= col_exec_count[col];
      } else {
        col_avg[col] = 0.0;
      }
    } 

    // Column standard deviaation...
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];          

      int col = 0;
      for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
        const FOMGroup& group = fom_groups[ifg];

        for (size_t gv = 0; gv < group.variants.size(); ++gv) {
          VariantID comp_vid = group.variants[gv];

          if ( kern->wasVariantRun(comp_vid) ) {
            col_stddev[col] += ( pct_diff[ik][col] - col_avg[col] ) *
                               ( pct_diff[ik][col] - col_avg[col] );
          } 

          col++;

        } // loop over group variants

      }  // loop over groups

    }  // loop over kernels
 
    for (size_t col = 0; col < ncols; ++col) {
      if ( col_exec_count[col] > 0 ) {
        col_stddev[col] /= col_exec_count[col];
      } else {
        col_stddev[col] = 0.0;
      }
    }

    // 
    // Print column summaries.
    // 
    file <<left<< setw(kercol_width) << " ";
    for (size_t iv = 0; iv < ncols; ++iv) {
      file << sepchr << setw(fom_col_width) <<left<< "  " <<right<< pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Min";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_min[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Max";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_max[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Avg";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_avg[col] << pass;
    }
    file << endl;

    file <<left<< setw(kercol_width) << "Col Std Dev";
    for (size_t col = 0; col < ncols; ++col) {
      file << sepchr <<left<< setw(fom_col_width) << setprecision(prec) 
           << col_stddev[col] << pass;
    }
    file << endl;

    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


void Executor::writeChecksumReport(const string& filename)
{
  ofstream file(filename.c_str(), ios::out | ios::trunc);
  if ( !file ) {
    cout << " ERROR: Can't open output file " << filename << endl;
  }

  if ( file ) {

    //
    // Set basic table formatting parameters.
    //
    const string equal_line("===================================================================================================");
    const string dash_line("----------------------------------------------------------------------------------------");
    const string dash_line_short("-------------------------------------------------------");
    string dot_line("........................................................");

    size_t prec = 20;
    size_t checksum_width = prec + 8;

    size_t namecol_width = 0;
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      namecol_width = max(namecol_width, kernels[ik]->getName().size()); 
    }
    for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
      namecol_width = max(namecol_width, 
                          getVariantName(variant_ids[iv]).size()); 
    }
    namecol_width++;


    //
    // Print title.
    //
    file << equal_line << endl;
    file << "Checksum Report " << endl;
    file << equal_line << endl;

    //
    // Print column title line.
    //
    file <<left<< setw(namecol_width) << "Kernel  " << endl;
    file << dot_line << endl;
    file <<left<< setw(namecol_width) << "Variants  " 
         <<left<< setw(checksum_width) << "Checksum  " 
         <<left<< setw(checksum_width) 
         << "Checksum Diff (vs. first variant listed)"; 
    file << endl;
    file << dash_line << endl;

    //
    // Print checksum and diff against baseline for each kernel variant.
    //
    for (size_t ik = 0; ik < kernels.size(); ++ik) {
      KernelBase* kern = kernels[ik];

      file <<left<< setw(namecol_width) << kern->getName() << endl;
      file << dot_line << endl;

      Checksum_type cksum_ref = 0.0;
      size_t ivck = 0;
      bool found_ref = false;
      while ( ivck < variant_ids.size() && !found_ref ) {
        VariantID vid = variant_ids[ivck];
        if ( kern->wasVariantRun(vid) ) {
          cksum_ref = kern->getChecksum(vid);
          found_ref = true;
        }
        ++ivck;
      }

      for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
        VariantID vid = variant_ids[iv];
 
        if ( kern->wasVariantRun(vid) ) {
          Checksum_type vcheck_sum = kern->getChecksum(vid);
          Checksum_type diff = cksum_ref - kern->getChecksum(vid);

          file <<left<< setw(namecol_width) << getVariantName(vid)
               << showpoint << setprecision(prec) 
               <<left<< setw(checksum_width) << vcheck_sum
               <<left<< setw(checksum_width) << diff << endl;
        } else {
          file <<left<< setw(namecol_width) << getVariantName(vid) 
               <<left<< setw(checksum_width) << "Not Run" 
               <<left<< setw(checksum_width) << "Not Run" << endl;
        }

      }

      file << endl;
      file << dash_line_short << endl;
    }
    
    file.flush(); 

  } // note file will be closed when file stream goes out of scope
}


string Executor::getReportTitle(CSVRepMode mode)
{
  string title;
  switch ( mode ) {
    case CSVRepMode::Timing : { 
      title = string("Mean Runtime Report (sec.) "); 
      break; 
    }
    case CSVRepMode::Speedup : { 
      if ( haveReferenceVariant() ) {
        title = string("Speedup Report (T_ref/T_var)") +
                string(": ref var = ") + getVariantName(reference_vid) + 
                string(" ");
      }
      break; 
    }
    default : { cout << "\n Unknown CSV report mode = " << mode << endl; }
  }; 
  return title;
}

long double Executor::getReportDataEntry(CSVRepMode mode,
                                         KernelBase* kern, 
                                         VariantID vid)
{
  long double retval = 0.0; 
  switch ( mode ) {
    case CSVRepMode::Timing : { 
      retval = kern->getTotTime(vid) / run_params.getNumPasses();
      break; 
    }
    case CSVRepMode::Speedup : { 
      if ( haveReferenceVariant() ) {
        if ( kern->hasVariantToRun(reference_vid) && 
             kern->hasVariantToRun(vid) ) {
          retval = kern->getTotTime(reference_vid) / kern->getTotTime(vid);
        } else {
          retval = 0.0;
        }
#if 0 // RDH DEBUG  (leave this here, it's useful for debugging!)
        cout << "Kernel(iv): " << kern->getName() << "(" << vid << ")" << endl;
        cout << "\tref_time, tot_time, retval = " 
             << kern->getTotTime(reference_vid) << " , "
             << kern->getTotTime(vid) << " , "
             << retval << endl;
#endif
      }
      break; 
    }
    default : { cout << "\n Unknown CSV report mode = " << mode << endl; }
  }; 
  return retval;
}

void Executor::getFOMGroups(vector<FOMGroup>& fom_groups)
{
  fom_groups.clear();

  for (size_t iv = 0; iv < variant_ids.size(); ++iv) {
    VariantID vid = variant_ids[iv];
    string vname = getVariantName(vid);

    if ( vname.find("Base") != string::npos ) {

      FOMGroup group;
      group.base = vid;
 
      string::size_type pos = vname.find("_");
      string pm(vname.substr(pos+1, string::npos));

      for (size_t ivs = iv+1; ivs < variant_ids.size(); ++ivs) {
        VariantID vids = variant_ids[ivs];
        if ( getVariantName(vids).find(pm) != string::npos ) {
          group.variants.push_back(vids);
        }
      }

      if ( !group.variants.empty() ) {
        fom_groups.push_back( group );
      }

    }  // if variant name contains 'Base'

  }  // iterate over variant ids to run

#if 0 //  RDH DEBUG   (leave this here, it's useful for debugging!)
  cout << "\nFOMGroups..." << endl;
  for (size_t ifg = 0; ifg < fom_groups.size(); ++ifg) {
    const FOMGroup& group = fom_groups[ifg];
    cout << "\tBase : " << getVariantName(group.base) << endl;
    for (size_t iv = 0; iv < group.variants.size(); ++iv) {
      cout << "\t\t " << getVariantName(group.variants[iv]) << endl;
    }
  }
#endif
}
// TODO:  AJP and DZP talk these functions through;
// is the arrow operator here acting as a pointer object to registerGroup, etc.?

void free_register_group(Executor* exec, std::string groupName){
   exec->registerGroup(groupName);
}
void free_register_kernel(Executor* exec, std::string groupName, KernelBase* kernel) {
   exec->registerKernel(groupName, kernel);
}
}  // closing brace for rajaperf namespace
