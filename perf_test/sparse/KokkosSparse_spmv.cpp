/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

#include <Kokkos_Core.hpp>
#include <matrix_market.hpp>

#include <KokkosKernels_SPMV.hpp>
#include <Kokkos_SPMV.hpp>
#include <Kokkos_SPMV_Inspector.hpp>
#include <CuSparse_SPMV.hpp>
#include <MKL_SPMV.hpp>

#ifdef _OPENMP
#include <OpenMPStatic_SPMV.hpp>
#include <OpenMPDynamic_SPMV.hpp>
#include <OpenMPSmartStatic_SPMV.hpp>
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_YAML
#include "yaml-cpp/yaml.h"
#include "Kokkos_Performance_impl.hpp"
#endif

enum {KOKKOS, MKL, CUSPARSE, KK_KERNELS, KK_KERNELS_INSP, KK_INSP, OMP_STATIC, OMP_DYNAMIC, OMP_INSP};
enum {AUTO, DYNAMIC, STATIC};

typedef Kokkos::DefaultExecutionSpace execution_space;

#ifdef INT64
typedef long long int LocalOrdinalType;
#else
typedef int LocalOrdinalType;
#endif


template< typename ScalarType , typename OrdinalType>
int SparseMatrix_generate(OrdinalType nrows, OrdinalType ncols, OrdinalType &nnz, OrdinalType varianz_nel_row, OrdinalType width_row, ScalarType* &values, OrdinalType* &rowPtr, OrdinalType* &colInd)
{
  rowPtr = new OrdinalType[nrows+1];

  OrdinalType elements_per_row = nnz/nrows;
  srand(13721);
  rowPtr[0] = 0;
  for(int row=0;row<nrows;row++)
  {
    int varianz = (1.0*rand()/INT_MAX-0.5)*varianz_nel_row;
    rowPtr[row+1] = rowPtr[row] + elements_per_row+varianz;
  }
  nnz = rowPtr[nrows];
  values = new ScalarType[nnz];
  colInd = new OrdinalType[nnz];
  for(int row=0;row<nrows;row++)
  {
         for(int k=rowPtr[row];k<rowPtr[row+1];k++)
         {
                int pos = (1.0*rand()/INT_MAX-0.5)*width_row+row;
                if(pos<0) pos+=ncols;
                if(pos>=ncols) pos-=ncols;
                colInd[k]= pos;
                values[k] = 100.0*rand()/INT_MAX-50.0;
         }
  }
  return nnz;
}

template<typename AType, typename XType, typename YType>
void matvec(AType& A, XType x, YType y, int rows_per_thread, int team_size, int vector_length, int test, int schedule) {

        switch(test) {

        case KOKKOS:
                if(schedule == AUTO)
                  schedule = A.nnz()>10000000?DYNAMIC:STATIC;
                if(schedule == STATIC)
                  kk_matvec<AType,XType,YType,Kokkos::Static>(A, x, y, rows_per_thread, team_size, vector_length);
                if(schedule == DYNAMIC)
                  kk_matvec<AType,XType,YType,Kokkos::Dynamic>(A, x, y, rows_per_thread, team_size, vector_length);
                break;
        case KK_INSP:
                if(schedule == AUTO)
                  schedule = A.nnz()>10000000?DYNAMIC:STATIC;
                if(schedule == STATIC)
                  kk_inspector_matvec<AType,XType,YType,Kokkos::Static>(A, x, y, rows_per_thread, team_size, vector_length);
                if(schedule == DYNAMIC)
                  kk_inspector_matvec<AType,XType,YType,Kokkos::Dynamic>(A, x, y, rows_per_thread, team_size, vector_length);
                break;

#ifdef _OPENMP
        case OMP_STATIC:
                openmp_static_matvec<AType, XType, YType, int, double>(A, x, y, rows_per_thread, team_size, vector_length);
                break;
        case OMP_DYNAMIC:
                openmp_dynamic_matvec<AType, XType, YType, int, double>(A, x, y, rows_per_thread, team_size, vector_length);
                break;
        case OMP_INSP:
                openmp_smart_static_matvec<AType, XType, YType, int, double>(A, x, y, rows_per_thread, team_size, vector_length);
                break;
#endif

#ifdef HAVE_MKL
        case MKL:
                mkl_matvec(A, x, y, rows_per_thread, team_size, vector_length);
                break;
#endif
#ifdef HAVE_CUSPARSE
        case CUSPARSE:
                cusparse_matvec(A, x, y, rows_per_thread, team_size, vector_length);
                break;
#endif
#ifdef HAVE_KK_KERNELS
        case KK_KERNELS:
                kokkoskernels_matvec(A, x, y, rows_per_thread, team_size, vector_length);
                break;
  case KK_KERNELS_INSP:
    if(A.graph.row_block_offsets.data()==NULL) {
      printf("PTR: %p\n",A.graph.row_block_offsets.data());
      A.graph.create_block_partitioning(AType::execution_space::concurrency());
      printf("PTR2: %p\n",A.graph.row_block_offsets.data());
    }
    kokkoskernels_matvec(A, x, y, rows_per_thread, team_size, vector_length);
    break;
#endif
        default:
                fprintf(stderr, "Selected test is not available.\n");

        }
}

template<typename Scalar>
int test_crs_matrix_singlevec(int numRows, int numCols, int nnz, int test, const char* filename, const bool binaryfile, int rows_per_thread, int team_size, int vector_length,int idx_offset, int schedule, int loop) {
  typedef KokkosSparse::CrsMatrix<Scalar,int,execution_space,void,int> matrix_type ;
  typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,execution_space> mv_type;
  typedef typename Kokkos::View<Scalar*,Kokkos::LayoutLeft,execution_space,Kokkos::MemoryRandomAccess > mv_random_read_type;
  typedef typename mv_type::HostMirror h_mv_type;

  Scalar* val = NULL;
  int* row = NULL;
  int* col = NULL;

  srand(17312837);
  if(filename==NULL)
    nnz = SparseMatrix_generate<Scalar,int>(numRows,numCols,nnz,nnz/numRows*0.2,numRows*0.01,val,row,col);
  else
    if(!binaryfile)
      nnz = SparseMatrix_MatrixMarket_read<Scalar,int>(filename,numRows,numCols,nnz,val,row,col,false,idx_offset);
    else
      nnz = SparseMatrix_ReadBinaryFormat<Scalar,int>(filename,numRows,numCols,nnz,val,row,col);

  matrix_type A("CRS::A",numRows,numCols,nnz,val,row,col,false);

  mv_type x("X",numCols);
  mv_random_read_type t_x(x);
  mv_type y("Y",numRows);
  h_mv_type h_x = Kokkos::create_mirror_view(x);
  h_mv_type h_y = Kokkos::create_mirror_view(y);
  h_mv_type h_y_compare = Kokkos::create_mirror(y);

  typename matrix_type::StaticCrsGraphType::HostMirror h_graph = Kokkos::create_mirror(A.graph);
  typename matrix_type::values_type::HostMirror h_values = Kokkos::create_mirror_view(A.values);

  for(int i=0; i<numCols;i++) {
    h_x(i) = (Scalar) (1.0*(rand()%40)-20.);
  }
  for(int i=0; i<numRows;i++) {
    h_y(i) = (Scalar) (1.0*(rand()%40)-20.);
  }

  // Error Check Gold Values
  for(int i=0;i<numRows;i++) {
    int start = h_graph.row_map(i);
    int end = h_graph.row_map(i+1);
    for(int j=start;j<end;j++) {
      h_values(j) = h_graph.entries(j) + i;
    }

    h_y_compare(i) = 0;
    for(int j=start;j<end;j++) {
      Scalar val = h_graph.entries(j) + i;
      int idx = h_graph.entries(j);
      h_y_compare(i)+=val*h_x(idx);
    }
  }

  Kokkos::deep_copy(x,h_x);
  Kokkos::deep_copy(y,h_y);
  Kokkos::deep_copy(A.graph.entries,h_graph.entries);
  Kokkos::deep_copy(A.values,h_values);
  typename KokkosSparse::CrsMatrix<Scalar,int,execution_space,void,int>::values_type x1("X1",numCols);
  Kokkos::deep_copy(x1,h_x);
  typename KokkosSparse::CrsMatrix<Scalar,int,execution_space,void,int>::values_type y1("Y1",numRows);

  int nnz_per_row = A.nnz()/A.numRows();
  matvec(A,x1,y1,rows_per_thread,team_size,vector_length,test,schedule);

  // Error Check
  Kokkos::deep_copy(h_y,y1);
  Scalar error = 0;
  Scalar sum = 0;
  for(int i=0;i<numRows;i++) {
    error += (h_y_compare(i)-h_y(i))*(h_y_compare(i)-h_y(i));
    sum += h_y_compare(i)*h_y_compare(i);
  }

  int num_errors = 0;
  double total_error = 0;
  double total_sum = 0;
  num_errors += (error/(sum==0?1:sum))>1e-5?1:0;
  total_error += error;
  total_sum += sum;

  // Benchmark
  double min_time = 1.0e32;
  double max_time = 0.0;
  double ave_time = 0.0;
  for(int i=0;i<loop;i++) {
    Kokkos::Timer timer;
    matvec(A,x1,y1,rows_per_thread,team_size,vector_length,test,schedule);
    execution_space::fence();
    double time = timer.seconds();
    ave_time += time;
    if(time>max_time) max_time = time;
    if(time<min_time) min_time = time;
  }

  // Performance Output
  double matrix_size = 1.0*((nnz*(sizeof(Scalar)+sizeof(int)) + numRows*sizeof(int)))/1024/1024;
  double vector_size = 2.0*numRows*sizeof(Scalar)/1024/1024;
  double vector_readwrite = (nnz+numCols)*sizeof(Scalar)/1024/1024;

  double problem_size = matrix_size+vector_size;
  double ave_bandwidth = (matrix_size+vector_readwrite)/ave_time*loop/1024;
  double min_bandwidth = (matrix_size+vector_readwrite)/max_time/1024;
  double max_bandwidth = (matrix_size+vector_readwrite)/min_time/1024;
  double ave_Gflop = 2.0*nnz*loop/ave_time/1e9;
  double min_Gflop = 2.0*nnz/max_time/1e9;
  double max_Gflop = 2.0*nnz/min_time/1e9;
  double ave_time_ms = ave_time/loop*1000;
  double max_time_ms = max_time*1000;
  double min_time_ms = min_time*1000;

  printf("NNZ NumRows NumCols ProblemSize(MB) AveBandwidth(GB/s) MinBandwidth(GB/s) MaxBandwidth(GB/s) AveGFlop MinGFlop MaxGFlop aveTime(ms) maxTime(ms) minTime(ms) numErrors\n");
  printf("%i %i %i %6.2lf ( %6.2lf %6.2lf %6.2lf ) ( %6.3lf %6.3lf %6.3lf ) ( %6.3lf %6.3lf %6.3lf ) %i RESULT\n",nnz, numRows,numCols,problem_size,
          ave_bandwidth, min_bandwidth, max_bandwidth,
          ave_Gflop, min_Gflop, max_Gflop,
          ave_time_ms, max_time_ms, min_time_ms,
          num_errors);

#ifdef KOKKOSKERNELS_ENABLE_TPL_YAML

  printf("Using Performance Archiver\n");

  // set up some user options
  std::string archiveName("KokkosSparse_spmv.yaml"); //  name of the archive
  std::string testName = "KokkosSparse_spmv";        // name of test
  std::string hostName;      // optional hostname - auto detected if blank

  // Make a schedule string - not sure if we want this in the yaml archive
  std::string schedule_string = "Unknown";
  switch(schedule) {
    case AUTO:    schedule_string = "AUTO";    break;
    case STATIC:  schedule_string = "STATIC";  break;
    case DYNAMIC: schedule_string = "DYNAMIC"; break;
  }

  // Make a test string
  std::string test_string = "Unknown";
  switch(test) {
    case KOKKOS:          test_string = "KOKKOS";          break;
    case KK_INSP:         test_string = "KK_INSP";         break;
#ifdef _OPENMP
    case OMP_STATIC:      test_string = "OMP_STATIC";      break;
    case OMP_DYNAMIC:     test_string = "OMP_DYNAMIC";     break;
    case OMP_INSP:        test_string = "OMP_INSP";        break;
#endif
#ifdef HAVE_MKL
    case MKL:             test_string = "MKL";             break;
#endif
#ifdef HAVE_CUSPARSE
    case CUSPARSE:        test_string = "CUSPARSE";        break;
#endif
#ifdef HAVE_KK_KERNELS
    case KK_KERNELS:      test_string = "KK_KERNELS";      break;
    case KK_KERNELS_INSP: test_string = "KK_KERNELS_INSP"; break;
#endif
  }

  using KokkosKernels::Performance;

  Performance archiver; // Create archiver

  // Add machine config options
  archiver.set_machine_config("Test", test_string);
  archiver.set_machine_config("Schedule", schedule_string);

  // Fill config
  archiver.set_config("nnz", nnz);
  archiver.set_config("numRows", numRows);
  archiver.set_config("numCols", numCols);
  archiver.set_config("filename", filename ? filename : ""); // null is problematic for current setup
  archiver.set_config("binaryfile", binaryfile);
  archiver.set_config("rows_per_thread", rows_per_thread);
  archiver.set_config("team_size", team_size);
  archiver.set_config("vector_length", vector_length);
  archiver.set_config("idx_offset", idx_offset);
  archiver.set_config("loop", loop);

  // Fill results
  double tolerance = 0.5; // probably to change or make specific to each value
  archiver.set_result("AveBandwidth(GB/s)", ave_bandwidth, tolerance);
  archiver.set_result("MinBandwidth(GB/s)", min_bandwidth, tolerance);
  archiver.set_result("MaxBandwidth(GB/s)", max_bandwidth, tolerance);
  archiver.set_result("AveGFlop",           ave_Gflop,     tolerance);
  archiver.set_result("MinGFlop",           min_Gflop,     tolerance);
  archiver.set_result("MaxGFlop",           max_Gflop,     tolerance);
  archiver.set_result("aveTime(ms)",        ave_time_ms,   tolerance);
  archiver.set_result("maxTime(ms)",        max_time_ms,   tolerance);
  archiver.set_result("minTime(ms)",        min_time_ms,   tolerance);
  archiver.set_result("numErrors",          num_errors);

  // run it
  Performance::Result result = archiver.run(archiveName, testName, hostName);

  // print the yaml file for inspection
  Performance::print_archive(archiveName);

  // Print results
  switch (result) {
    case Performance::Passed:
      std::cout << "Archiver Passed" << std::endl;
      break;
    case Performance::Failed:
      std::cout << "Archiver Failed" << std::endl;
      break;
    case Performance::NewMachine:
      std::cout << "Archiver Passed. Adding new machine entry." << std::endl;
      break;
    case Performance::NewConfiguration:
      std::cout << "Archiver Passed. Adding new machine configuration." << std::endl;
      break;
    case Performance::NewTest:
      std::cout << "Archiver Passed. Adding new test entry." << std::endl;
      break;
    case Performance::NewTestConfiguration:
      std::cout << "Archiver Passed. Adding new test entry configuration." << std::endl;
      break;
    case Performance::UpdatedTest:
      std::cout << "Archiver Passed. Updating test entry." << std::endl;
      break;
    default:
      throw std::logic_error("Unexpected result code.");
      break;
  }

#endif
  return (int)total_error;
}

void print_help() {
  printf("SPMV benchmark code written by Christian Trott.\n");
  printf("OpenMP implementations written by Simon Hammond (Sandia National Laboratories).\n\n");
  printf("Options:\n");
  printf("  -s [N]          : generate a semi-random banded (band size 0.01xN) NxN matrix\n");
  printf("                    with average of 10 entries per row.\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                      kk,kk-kernels          (Kokkos/Trilinos)\n");
  printf("                      kk-insp                (Kokkos Structure Inspection)\n");
#ifdef _OPENMP
  printf("                      omp-dynamic,omp-static (Standard OpenMP)\n");
  printf("                      omp-insp               (OpenMP Structure Inspection)\n");
#endif
  printf("                      mkl,cusparse           (Vendor Libraries)\n\n");
  printf("  --schedule [SCH]: Set schedule for kk variant (static,dynamic,auto [ default ]).\n");
  printf("  -f [file]       : Read in Matrix Market formatted text file 'file'.\n");
  printf("  -fb [file]      : Read in binary Matrix files 'file'.\n");
  printf("  --write-binary  : In combination with -f, generate binary files.\n");
  printf("  --offset [O]    : Subtract O from every index.\n");
  printf("                    Useful in case the matrix market file is not 0 based.\n\n");
  printf("  -rpt [K]        : Number of Rows assigned to a thread.\n");
  printf("  -ts [T]         : Number of threads per team.\n");
  printf("  -vl [V]         : Vector-length (i.e. how many Cuda threads are a Kokkos 'thread').\n");
  printf("  -l [LOOP]       : How many spmv to run to aggregate average time. \n");
}

int main(int argc, char **argv)
{
 long long int size = 110503; // a prime number
 int numVecs = 4;
 int test=KOKKOS;
 int type=-1;
 char* filename = NULL;
 bool binaryfile = false;
 bool write_binary = false;

 int rows_per_thread = -1;
 int vector_length = -1;
 int team_size = -1;
 int idx_offset = 0;
 int schedule=AUTO;
 int loop = 100;

 if(argc == 1) {
   print_help();
   return 0;
 }

 for(int i=0;i<argc;i++)
 {
  if((strcmp(argv[i],"-s")==0)) {size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-v")==0)) {numVecs=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--test")==0)) {
    i++;
    if((strcmp(argv[i],"mkl")==0))
      test = MKL;
    if((strcmp(argv[i],"kk")==0))
      test = KOKKOS;
    if((strcmp(argv[i],"cusparse")==0))
      test = CUSPARSE;
    if((strcmp(argv[i],"kk-kernels")==0))
      test = KK_KERNELS;
    if((strcmp(argv[i],"kk-kernels-insp")==0))
      test = KK_KERNELS_INSP;
    if((strcmp(argv[i],"kk-insp")==0))
      test = KK_INSP;
#ifdef _OPENMP
    if((strcmp(argv[i],"omp-static") == 0))
      test = OMP_STATIC;
    if((strcmp(argv[i], "omp-dynamic") == 0))
      test = OMP_DYNAMIC;
    if((strcmp(argv[i], "omp-insp") == 0))
      test = OMP_INSP;
#endif
    continue;
  }
  if((strcmp(argv[i],"--type")==0)) {type=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-f")==0)) {filename = argv[++i]; continue;}
  if((strcmp(argv[i],"-fb")==0)) {filename = argv[++i]; binaryfile = true; continue;}
  if((strcmp(argv[i],"-rpt")==0)) {rows_per_thread=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-ts")==0)) {team_size=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"-vl")==0)) {vector_length=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--offset")==0)) {idx_offset=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--write-binary")==0)) {write_binary=true;}
  if((strcmp(argv[i],"-l")==0)) {loop=atoi(argv[++i]); continue;}
  if((strcmp(argv[i],"--schedule")==0)) {
    i++;
    if((strcmp(argv[i],"auto")==0))
      schedule = AUTO;
    if((strcmp(argv[i],"dynamic")==0))
      schedule = DYNAMIC;
    if((strcmp(argv[i],"static")==0))
      schedule = STATIC;
    continue;
  }
  if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
    print_help();
    return 0;
  }
 }

 if(write_binary) {
   double* val = NULL;
   int* row = NULL;
   int* col = NULL;
   int numRows,numCols,nnz;
   SparseMatrix_WriteBinaryFormat<double,int>(filename,numRows,numCols,nnz,val,row,col,true,idx_offset);
   return 0;
 }

 Kokkos::initialize(argc,argv);

 int total_errors = test_crs_matrix_singlevec<double>(size,size,size*10,test,filename,binaryfile,rows_per_thread,team_size,vector_length,idx_offset,schedule,loop);

 if(total_errors == 0)
   printf("Kokkos::MultiVector Test: Passed\n");
 else
   printf("Kokkos::MultiVector Test: Failed\n");


  Kokkos::finalize();
}
