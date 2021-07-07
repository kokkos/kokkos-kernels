//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RunParams.hpp"

#include <cstdlib>
#include <cstdio>
#include <iostream>

namespace rajaperf
{

/*
 *******************************************************************************
 *
 * Ctor for PunParams class defines suite execution defaults and parses
 * command line args to set others that are specified when suite is run.
 *
 *******************************************************************************
 */
RunParams::RunParams(int argc, char** argv)
 : input_state(Undefined),
   show_progress(false),
   npasses(1),
   rep_fact(1.0),
   size_fact(1.0),
   pf_tol(0.1),
   checkrun_reps(1),
   size_spec(Specundefined),
   size_spec_string("SPECUNDEFINED"),
   reference_variant(),
   kernel_input(),
   invalid_kernel_input(),
   variant_input(),
   invalid_variant_input(),
   outdir(),
   outfile_prefix("RAJAPerf")
{
  parseCommandLineOptions(argc, argv);
}


/*
 *******************************************************************************
 *
 * Dtor for RunParams class.
 *
 *******************************************************************************
 */
RunParams::~RunParams()
{
}


/*
 *******************************************************************************
 *
 * Print all run params data to given output stream.
 *
 *******************************************************************************
 */
void RunParams::print(std::ostream& str) const
{
  str << "\n show_progress = " << show_progress; 
  str << "\n npasses = " << npasses; 
  str << "\n rep_fact = " << rep_fact; 
  str << "\n size_fact = " << size_fact; 
  str << "\n pf_tol = " << pf_tol; 
  str << "\n checkrun_reps = " << checkrun_reps; 
  str << "\n size_spec_string = " << size_spec_string;  
  str << "\n reference_variant = " << reference_variant; 
  str << "\n outdir = " << outdir; 
  str << "\n outfile_prefix = " << outfile_prefix; 

  str << "\n kernel_input = "; 
  for (size_t j = 0; j < kernel_input.size(); ++j) {
    str << "\n\t" << kernel_input[j];
  }
  str << "\n invalid_kernel_input = ";
  for (size_t j = 0; j < invalid_kernel_input.size(); ++j) {
    str << "\n\t" << invalid_kernel_input[j];
  }

  str << "\n variant_input = "; 
  for (size_t j = 0; j < variant_input.size(); ++j) {
    str << "\n\t" << variant_input[j];
  }
  str << "\n invalid_variant_input = "; 
  for (size_t j = 0; j < invalid_variant_input.size(); ++j) {
    str << "\n\t" << invalid_variant_input[j];
  }

  str << std::endl;
  str.flush();
}


/*
 *******************************************************************************
 *
 * Parse command line args to set how suite will run.
 *
 *******************************************************************************
 */
void RunParams::parseCommandLineOptions(int argc, char** argv)
{
  std::cout << "\n\nReading command line input..." << std::endl;

  for (int i = 1; i < argc; ++i) {

    std::string opt(argv[i]);

    if ( opt == std::string("--help") ||
         opt == std::string("-h") ) {

      printHelpMessage(std::cout);
      input_state = InfoRequest;

    } else if ( opt == std::string("--show-progress") ||
                opt == std::string("-sp") ) {

      show_progress = true;

    } else if ( opt == std::string("--print-kernels") ||
                opt == std::string("-pk") ) {
     
      printFullKernelNames(std::cout);     
      input_state = InfoRequest;
 
    } else if ( opt == std::string("--print-variants") ||
                opt == std::string("-pv") ) {

      printVariantNames(std::cout);     
      input_state = InfoRequest;
 
    } else if ( opt == std::string("--npasses") ) {

      i++;
      if ( i < argc ) { 
        npasses = ::atoi( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --npasses a value for number of passes (int)" 
                  << std::endl; 
        input_state = BadInput;
      }

    } else if ( opt == std::string("--repfact") ) {

      i++;
      if ( i < argc ) { 
        rep_fact = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --rep_fact a value (double)" 
                  << std::endl;       
        input_state = BadInput;
      }

    } else if ( opt == std::string("--sizefact") ) {

      i++;
      if ( i < argc ) { 
        size_fact = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --sizefact a value (double)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if (opt == std::string("--sizespec") ) {
      i++;
      if ( i < argc ) {
        setSizeSpec(argv[i]);
      } else {
        std::cout << "\nBad input:"
                  << " must give --sizespec a value for size specification: one of  MINI,SMALL,MEDIUM,LARGE,EXTRALARGE (string : any case)"
                  << std::endl;
        input_state = BadInput;
      }
    } else if ( opt == std::string("--pass-fail-tol") ||
                opt == std::string("-pftol") ) {

      i++;
      if ( i < argc ) {
        pf_tol = ::atof( argv[i] );
      } else {
        std::cout << "\nBad input:"
                  << " must give --pass-fail-tol (or -pftol) a value (double)"
                  << std::endl;
        input_state = BadInput;
      }

    } else if ( opt == std::string("--kernels") ||
                opt == std::string("-k") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          kernel_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--variants") ||
                std::string(argv[i]) == std::string("-v") ) {

      bool done = false;
      i++;
      while ( i < argc && !done ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
          done = true;
        } else {
          variant_input.push_back(opt);
          ++i;
        }
      }

    } else if ( std::string(argv[i]) == std::string("--outdir") ||
                std::string(argv[i]) == std::string("-od") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          outdir = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--outfile") ||
                std::string(argv[i]) == std::string("-of") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          outfile_prefix = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--refvar") ||
                std::string(argv[i]) == std::string("-rv") ) {

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          reference_variant = std::string( argv[i] );
        }
      }

    } else if ( std::string(argv[i]) == std::string("--dryrun") ) {

       input_state = DryRun;
   
    } else if ( std::string(argv[i]) == std::string("--checkrun") ) {

      input_state = CheckRun; 

      i++;
      if ( i < argc ) {
        opt = std::string(argv[i]);
        if ( opt.at(0) == '-' ) {
          i--;
        } else {
          checkrun_reps = ::atoi( argv[i] );
        }

      }

    } else {
     
      input_state = BadInput;

      std::string huh(argv[i]);   
      std::cout << "\nUnknown option: " << huh << std::endl;
      std::cout.flush();

    }

  }
}


void RunParams::printHelpMessage(std::ostream& str) const
{
  str << "\nUsage: ./raja-perf.exe [options]\n";
  str << "Valid options are:\n"; 

  str << "\t --help, -h (print options with descriptions}\n\n";

  str << "\t --show-progress, -sp (print progress during run}\n\n";

  str << "\t --print-kernels, -pk (print valid kernel names}\n\n";

  str << "\t --print-variants, -pv (print valid variant names}\n\n";

  str << "\t --npasses <int> [default is 1]\n"
      << "\t      (num passes through suite)\n"; 
  str << "\t\t Example...\n"
      << "\t\t --npasses 2 (runs complete suite twice\n\n";

  str << "\t --repfact <double> [default is 1.0]\n"
      << "\t      (fraction of default # reps to run each kernel)\n";
  str << "\t\t Example...\n"
      << "\t\t --repfact 0.5 (runs kernels 1/2 as many times as default)\n\n";

  str << "\t --sizefact <double> [default is 1.0]\n"
      << "\t      (fraction of default kernel iteration space size to run)\n";
  str << "\t\t Example...\n"
      << "\t\t --sizefact 2.0 (iteration space size is twice the default)\n\n";

  str << "\t --sizespec <string> [one of : mini,small,medium,large,extralarge (anycase) -- default is medium]\n"
      << "\t      (used to set specific sizes for polybench kernels)\n\n"; 

  str << "\t --pass-fail-tol, -pftol <double> [default is 0.1; i.e., 10%]\n"
      << "\t      (slowdown tolerance for RAJA vs. Base variants in FOM report)\n";
  str << "\t\t Example...\n"
      << "\t\t -pftol 0.2 (RAJA kernel variants that run 20% or more slower than Base variants will be reported as OVER_TOL in FOM report)\n\n";

  str << "\t --outdir, -od <string> [Default is current directory]\n"
      << "\t      (directory path for output data files)\n";
  str << "\t\t Examples...\n"
      << "\t\t --outdir foo (output files to ./foo directory\n"
      << "\t\t -od /nfs/tmp/me (output files to /nfs/tmp/me directory)\n\n";

  str << "\t --outfile, -of <string> [Default is RAJAPerf]\n"
      << "\t      (file name prefix for output files)\n";
  str << "\t\t Examples...\n"
      << "\t\t --outfile mydata (output data will be in files 'mydata*')\n"
      << "\t\t -of dat (output data will be in files 'dat*')\n\n";

  str << "\t --kernels, -k <space-separated strings> [Default is run all]\n"
      << "\t      (names of individual kernels and/or groups of kernels to run)\n"; 
  str << "\t\t Examples...\n"
      << "\t\t --kernels Polybench (run all kernels in Polybench group)\n"
      << "\t\t -k INIT3 MULADDSUB (run INIT3 and MULADDSUB kernels\n"
      << "\t\t -k INIT3 Apps (run INIT3 kernsl and all kernels in Apps group)\n\n";

  str << "\t --variants, -v <space-separated strings> [Default is run all]\n"
      << "\t      (names of variants)\n"; 
  str << "\t\t Examples...\n"
      << "\t\t -variants RAJA_CUDA (run RAJA_CUDA variants)\n"
      << "\t\t -v Base_Seq RAJA_CUDA (run Base_Seq, RAJA_CUDA variants)\n\n";

  str << "\t --refvar, -rv <string> [Default is none]\n"
      << "\t      (reference variant for speedup calculation)\n\n";
  str << "\t\t Example...\n"
      << "\t\t -refvar Base_Seq (speedups reported relative to Base_Seq variants)\n\n";

  str << "\t --checkrun <int> [default is 1]\n"
<< "\t      (run each kernel given number of times; usually to check things are working)\n"; 
  str << "\t\t Example...\n"
      << "\t\t --checkrun 2 (run each kernel twice)\n\n";

  str << "\t --dryrun (print summary of how suite will run without running)\n\n";

  str << std::endl;
  str.flush();
}


void RunParams::printKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels:";
  str << "\n------------------\n";
  // TODO: replace this logic
  //for (int ik = 0; ik < NumKernels; ++ik) {
///// RDH DISABLE COUPLE KERNEL
  //  if ( /** static_cast<KernelID>(ik) != Apps_COUPLE*/ true) {
  //    str << getKernelName(static_cast<KernelID>(ik)) << std::endl;
  //  }
  //}
  str.flush();
}


void RunParams::printFullKernelNames(std::ostream& str) const
{
  str << "\nAvailable kernels (<group name>_<kernel name>):";
  str << "\n-----------------------------------------\n";
  // TODO: replace
  //for (int ik = 0; ik < NumKernels; ++ik) {
///// RDH DISABLE COUPLE KERNEL
  //  if ( /** static_cast<KernelID>(ik) != Apps_COUPLE */ true) {
  //    str << getFullKernelName(static_cast<KernelID>(ik)) << std::endl;
  //  }
  //}
  str.flush();
}


void RunParams::printVariantNames(std::ostream& str) const
{
  str << "\nAvailable variants:";
  str << "\n-------------------\n";
  //for (int iv = 0; iv < NumVariants; ++iv) {
  //  str << getVariantName(static_cast<VariantID>(iv)) << std::endl;
  //}
  str.flush();
}


void RunParams::printGroupNames(std::ostream& str) const
{
  str << "\nAvailable groups:";
  str << "\n-----------------\n";
  //for (int is = 0; is < NumGroups; ++is) {
  //  str << getGroupName(static_cast<GroupID>(is)) << std::endl;
  //}
  str.flush();
}

const std::string& RunParams::getSizeSpecString()
{
  switch(size_spec) {
    case Mini:
      size_spec_string = "MINI";
      break;
    case Small:
      size_spec_string = "SMALL";
      break;
    case Medium:
      size_spec_string = "MEDIUM";
      break;
    case Large:
      size_spec_string = "LARGE";
      break;
    case Extralarge:
      size_spec_string = "EXTRALARGE";
      break;
    default:
      size_spec_string = "SPECUNDEFINED";
  }
  return size_spec_string;
}

void RunParams::setSizeSpec(std::string inputString)
{
  for (auto & c: inputString) c = std::toupper(c);
  if (inputString == "MINI")
    size_spec = Mini;
  else if (inputString == "SMALL")
    size_spec = Small;
  else if (inputString == "MEDIUM")
    size_spec = Medium;
  else if (inputString == "LARGE")
    size_spec = Large;
  else if (inputString == "EXTRALARGE")
    size_spec = Extralarge;
  else
    size_spec = Specundefined;
  std::cout << "Size Specification : " << getSizeSpecString() << std::endl;
}

}  // closing brace for rajaperf namespace
