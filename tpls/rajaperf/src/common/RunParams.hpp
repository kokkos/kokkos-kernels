//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJAPerf_RunParams_HPP
#define RAJAPerf_RunParams_HPP

#include <string>
#include <vector>
#include <iosfwd>

#include "RAJAPerfSuite.hpp"

namespace rajaperf
{

/*!
 *******************************************************************************
 *
 * \brief Simple class to parse and maintain suite execution parameters.
 *
 *******************************************************************************
 */
class RunParams {

public:
  RunParams( int argc, char** argv );
  ~RunParams( );

  /*!
   * \brief Enumeration indicating state of input options requested
   */
  enum InputOpt {
    InfoRequest,  /*!< option requesting information */
    DryRun,       /*!< report summary of how suite will run w/o running */
    CheckRun,     /*!< run suite with small rep count to make sure 
                       everything works properly */
    PerfRun,      /*!< input defines a valid performance run, 
                       suite will run as specified */
    BadInput,     /*!< erroneous input given */ 
    Undefined     /*!< input not defined (yet) */
  };

//@{
//! @name Methods to get/set input state

  InputOpt getInputState() const { return input_state; } 

  /*!
   * \brief Set whether run parameters (from input) are valid.
   */
  void setInputState(InputOpt is) { input_state = is; }

//@}


//@{
//! @name Getters/setters for processing input and run parameters

  bool showProgress() const { return show_progress; }

  int getNumPasses() const { return npasses; }

  double getRepFactor() const { return rep_fact; }

  double getSizeFactor() const { return size_fact; }

  SizeSpec  getSizeSpec() const { return size_spec; }

  void  setSizeSpec(std::string inputString);

  const std::string& getSizeSpecString();

  double getPFTolerance() const { return pf_tol; }

  int getCheckRunReps() const { return checkrun_reps; }

  const std::string& getReferenceVariant() const { return reference_variant; }

  const std::vector<std::string>& getKernelInput() const 
                                  { return kernel_input; }
  void setInvalidKernelInput( std::vector<std::string>& svec )
                              { invalid_kernel_input = svec; }
  const std::vector<std::string>& getInvalidKernelInput() const
                                  { return invalid_kernel_input; }

  const std::vector<std::string>& getVariantInput() const 
                                  { return variant_input; }
  void setInvalidVariantInput( std::vector<std::string>& svec )
                               { invalid_variant_input = svec; }
  const std::vector<std::string>& getInvalidVariantInput() const
                                  { return invalid_variant_input; }

  const std::string& getOutputDirName() const { return outdir; }
  const std::string& getOutputFilePrefix() const { return outfile_prefix; }

//@}

  /*!
   * \brief Print all run params data to given output stream.
   */
  void print(std::ostream& str) const;


private:
  RunParams() = delete;

//@{
//! @name Routines used in command line parsing
  void parseCommandLineOptions(int argc, char** argv);
  void printHelpMessage(std::ostream& str) const;
  void printFullKernelNames(std::ostream& str) const;
  void printKernelNames(std::ostream& str) const;
  void printVariantNames(std::ostream& str) const;
  void printGroupNames(std::ostream& str) const;
//@}

  InputOpt input_state;  /*!< state of command line input */

  bool show_progress;    /*!< true -> show run progress; false -> do not */

  int npasses;           /*!< Number of passes through suite  */
  double rep_fact;       /*!< pct of default kernel reps to run */
  double size_fact;      /*!< pct of default kernel iteration space to run */
  double pf_tol;         /*!< pct RAJA variant run time can exceed base for
                              each PM case to pass/fail acceptance */

  int checkrun_reps;     /*!< Num reps each kernel is run in check run */

  SizeSpec size_spec;    /*!< optional use/parse polybench spec file for size */

  std::string size_spec_string;

  std::string reference_variant;   /*!< Name of reference variant for speedup
                                        calculations */ 

  //
  // Arrays to hold input strings for valid/invalid input. Helpful for  
  // debugging command line args.
  //
  std::vector<std::string> kernel_input;
  std::vector<std::string> invalid_kernel_input;
  std::vector<std::string> variant_input;
  std::vector<std::string> invalid_variant_input;

  std::string outdir;          /*!< Output directory name. */
  std::string outfile_prefix;  /*!< Prefix for output data file names. */

};


}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
