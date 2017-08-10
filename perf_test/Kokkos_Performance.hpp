// @HEADER
// ***********************************************************************
//
//                    KokkosKernels: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef KOKKOS_PERFORMANCE_HPP
#define KOKKOS_PERFORMANCE_HPP

#include "yaml-cpp/yaml.h"

namespace KokkosKernels {

class Performance {
  public:
    /**
     * \brief Performance class manages the archive.
     * The pattern is create, add config, times, results, then run
     *
     * This will generate a starting point for a machine configuration. Users
     * should add new entries via set_machine_config. For example Kokkos users
     * might want to provide the name of the user Kokkos NodeType or Kokkos
     * DeviceType. Contains information mostly extracted from /proc/cpuinfo if
     * possible. On non unix systems most values will be unknown. Entries are:
     * - Compiler: The compiler name.
     * - Compiler_Version: A compiler version number.
     * - CPU_Name: The CPUs model name.
     * - CPU_Sockets: Number of CPU sockets in the system.
     * - CPU_Cores_Per_Socket: Number of CPU cores per socket.
     * - CPU_Total_HyperThreads: Total number of threads in a node.
     */
    Performance() : machine_configuration_node(get_machine_configuration()) {}

    /**
     * \brief set_config adds/changes a test config parameter
     *
     * \param name [in] The name used to identify the parameter in the archive.
     * \param val [in] The value assinged to the parameter.
     */
    template<typename T>
    void set_config(const std::string& name, const T& val) {
      test_configuration_node[name] = val;
    }

    /**
     * \brief set_result adds/changes a test result with a tolerance
     *
     * \param name [in] The name used to identify the result.
     * \param val [in] The recorded result which came from running the test.
     * \param tolerance [in] abs((new-old)/old)>tolerance triggers test failure.
     */
    void set_result(const std::string& name, double val, double tolerance) {
      validate_input_result_name(name);
      results_node[name] =
        Performance::Tolerance(val, tolerance).as_string();
    }

    /**
     * \brief set_result adds/changes a test result
     *
     * \param name [in] The name used to identify the result.
     * \param time [in] The recorded time which came from running the test.
     * \param tolerance_low [in] Lower bound of tolerance.
     * \param tolerance_high [in] Upper bound of tolerance.
     */
    void set_result(const std::string& name, double val,
      double tolerance_low, double tolerance_high) {
      validate_input_result_name(name);
      results_node[name] =
        Performance::Tolerance(val, tolerance_low, tolerance_high).as_string();
    }

    /**
     * \brief set_result adds/changes a test result for exact comparison
     *
     * \param name [in] The name used to identify the result.
     * \param val [in] The recorded result which came from running the test.
     */
    template<typename T>
    void set_result(const std::string& name, const T& val) {
      validate_input_result_name(name);
      results_node[mark_name_with_exact_code(name)] = val;
    }

    /**
     * \brief set_machine_config adds/changes a machine configuration
     *
     * \param name [in] The name used to identify the machine config parameter.
     * \param val [in] The setting for the machine config parameter.
     */
    template<typename T>
    void set_machine_config(const std::string& name, const T& val) {
      machine_configuration_node[name] = val;
    }

    /**
     * \brief Result codes after creating/comparing a test entry
     */
    enum Result{
      Failed,
      Passed,
      NewMachine,
      NewConfiguration,
      NewTest,
      NewTestConfiguration,
      UpdatedTest,
      Unknown};

    /**
     * \brief Processes the test and update the yaml archive
     * This should be called after all necessary inserts, such as set_config,
     * set_time, and set_result.
     *
     * \param test_name [in] Named used to match test entries.
     * \param archive_name [in] The local yaml path to generate the archive.
     * \param host_name [in] An optional hostname to be used instead of
     *   the one provided by the OS.
     *
     * \return Whether a matching test is found, or if it was added to an
     *   archive.
     *
     * Will search for a matching machine name with matching machine
     * configuration and matching test configuration. If one is found the
     * result values will be compared, if not a new test entry is
     * generated and the result written back to the file.
     *
     * Here is the list of valid return values:
     *
     * - Failed: Matching configuration found, but results are
     *     deviating more than the allowed tolerance.
     * - Passed: Matching configuration found, and results are
     *     within tolerances.
     * - NewMachine: The test archive didn't contain an entry with
     *     the same machine name. A new entry was generated.
     * - NewConfiguration: No matching machine configuration was
     *     found. A new entry was generated.
     * - NewTest: No matching testname was found. A new entry was
     *     generated and added to the archive.
     * - NewTestConfiguration: A matching testname was found, but
     *     different parameters were used. A new entry was generated.
     * - UpdatedTest: A matching test was found but more result
     *     values were given then previously found. The entry is updated.
     *     This will only happen if all the old result values are present in
     *     the new ones, and are within their respective tolerances.
     */
    Result run(const std::string& archive_name, const std::string& test_name,
      const std::string& host_name = "") const;

    /**
     * \brief print_archive will std::cout the yaml archive for inspection.
     * \param archive_name [in] The local yaml path to print.
     */
    static void print_archive(const std::string& archive_name);

    /**
     * \brief erase_archive will delete the archive.
     * \param archive_name [in] The local yaml path to print.
     */
    static void erase_archive(const std::string& archive_name);

  private:
    typedef YAML::Node node_t;

    /* Tolerance is an internal helper struct with a tuple of value and tolerance.
     * The tolerance can be either expressed as a relative or through an upper and
     * lower bound. This is now private to the Performance class and copies the
     * original Teuchos implementation.
     */
    struct Tolerance {
      double value;
      double lower;
      double upper;
      double tolerance;
      bool use_tolerance;
      Tolerance();
      Tolerance(double val, double tol);
      Tolerance(double val, double low, double up);
      Tolerance(std::string str);
      bool operator ==(const Tolerance& rhs);
      std::string as_string();
      void from_string(const std::string& valtol_str);
    };

    // Set up the machine configuration - users can modify the default setup
    node_t get_machine_configuration() const;

    // Compares two nodes and determines if they are have the same
    // members but does not compare the values of those members.
    bool hasSameElements(const node_t& a, const node_t& b) const;

    // does string include the exact key character '*'
    bool string_includes_exact_code(const std::string& name) const;

    // append the '*' key character to the name
    std::string mark_name_with_exact_code(const std::string& name) const;

    // make sure the input name doesn't have the character '*'
    void validate_input_result_name(const std::string& name) const;

    // private data members
    node_t machine_configuration_node; // stores machine config settings
    node_t test_configuration_node;    // stores test config settings
    node_t results_node;               // stores result settings
};

} // namespace KokkosKernels

#endif // KOKKOS_PERFORMANCE_HPP