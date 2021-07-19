/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jennifer Loe (jloe@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include<math.h>
#include"KokkosKernels_IOUtils.hpp"
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosSparse_spmv.hpp>

// This struct is returned to the user to give solver
// statistics and convergence status. 
struct GmresStats {
  int numIters;
  double endRelRes;
  enum FLAG { Conv, NoConv, LOA };
  FLAG convFlagVal;
  std::string convFlag() {
    switch(convFlagVal){
      case Conv:
        return "Converged";
      case NoConv:
        return "Not Converged";
      case LOA:
        return "Solver has had loss of accuracy.";
      default:
        return "Flag not defined.";
    }
  }
};

template< class ScalarType, class Layout, class EXSP, class OrdinalType = int > 
  GmresStats gmres( const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, EXSP> &A, 
                    const Kokkos::View<ScalarType*, Layout, EXSP> &B,
                    Kokkos::View<ScalarType*, Layout, EXSP> &X, 
                    const typename Kokkos::Details::ArithTraits<ScalarType>::mag_type tol = 1e-8, 
                    const int m=50, 
                    const int maxRestart=50, 
                    const std::string ortho = "CGS2"){

  typedef Kokkos::Details::ArithTraits<ScalarType> AT;
  typedef typename AT::val_type ST; // So this code will run with ScalarType = std::complex<T>.
  typedef typename AT::mag_type MT; 
  ST one = AT::one();
  ST zero = AT::zero();

  typedef Kokkos::View<ST*, Layout, EXSP> ViewVectorType;
  typedef Kokkos::View<ST*, Kokkos::LayoutRight, Kokkos::HostSpace> ViewHostVectorType; 
  typedef Kokkos::View<ST**, Layout, EXSP> ViewMatrixType;

  unsigned int n = A.numRows();

  // Check compatibility of dimensions at run time.
  if ( n != unsigned(A.numCols()) ){
    std::ostringstream os;
    os << "gmres: A must be a square matrix: " 
      << "numRows: " << n << "  numCols: " << A.numCols();
      Kokkos::Impl::throw_runtime_exception (os.str ());
  }

  if (X.extent(0) != B.extent(0) ||
      X.extent(0) != n ) {
    std::ostringstream os;
    os << "gmres: Dimensions of A, X, and B do not match: "
       << "A: " << n << " x " << n << ", X: " << X.extent(0) 
       << "x 1, B: " << B.extent(0) << " x 1";
    Kokkos::Impl::throw_runtime_exception (os.str ());
  }
  //Check parameter validity:
  if(m <= 0){
    throw std::invalid_argument("gmres: please choose restart size m greater than zero.");
  }
  if(maxRestart < 0){
    throw std::invalid_argument("gmres: Please choose maxRestart greater than zero.");
  }

  bool converged = false;
  int cycle = 0; // How many times have we restarted? 
  int numIters = 0;  //Number of iterations within the cycle before convergence.
  MT nrmB, trueRes, relRes, shortRelRes;
  GmresStats myStats;
  
  std::cout << "Convergence tolerance is: " << tol << std::endl;

  ViewVectorType Xiter("Xiter",n); //Intermediate solution at iterations before restart. 
  ViewVectorType Res(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Res"),n); //Residual vector
  ViewVectorType Wj(Kokkos::view_alloc(Kokkos::WithoutInitializing, "W_j"),n); //Tmp work vector 1
  ViewHostVectorType GVec_h(Kokkos::view_alloc(Kokkos::WithoutInitializing, "GVec"),m+1);
  ViewMatrixType GLsSoln("GLsSoln",m,1);//LS solution vec for Givens Rotation. Must be 2-D for trsm. 
  typename ViewMatrixType::HostMirror GLsSoln_h = Kokkos::create_mirror_view(GLsSoln); //This one is needed for triangular solve. 
  ViewHostVectorType CosVal_h("CosVal",m);
  ViewHostVectorType SinVal_h("SinVal",m);
  ViewMatrixType V(Kokkos::view_alloc(Kokkos::WithoutInitializing, "V"),n,m+1);
  ViewMatrixType VSub; //Subview of 1st m cols for updating soln. 

  ViewMatrixType H("H",m+1,m); //H matrix on device. Also used in Arn Rec debug. 
  typename ViewMatrixType::HostMirror H_h = Kokkos::create_mirror_view(H); //Make H into a host view of H. 

  //Compute initial residuals:
  nrmB = KokkosBlas::nrm2(B);
  Kokkos::deep_copy(Res,B);
  KokkosSparse::spmv("N", one, A, X, zero, Wj); // wj = Ax
  KokkosBlas::axpy(-one, Wj, Res); // res = res-Wj = b-Ax. 
  trueRes = KokkosBlas::nrm2(Res);
  relRes = trueRes/nrmB;
  shortRelRes = relRes;
    
  while( !converged && cycle <= maxRestart){
    GVec_h(0) = trueRes;

    // Run Arnoldi iteration:
    auto Vj = Kokkos::subview(V,Kokkos::ALL,0); 
    Kokkos::deep_copy(Vj,Res);
    KokkosBlas::scal(Vj,one/trueRes,Vj); //V0 = V0/norm(V0)

    for (int j = 0; j < m; j++){
      KokkosSparse::spmv("N", one, A, Vj, zero, Wj); //wj = A*Vj
      if( ortho == "MGS"){
        for (int i = 0; i <= j; i++){
          auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
          H_h(i,j) = KokkosBlas::dot(Vi,Wj);  //Vi^* Wj  
          KokkosBlas::axpy(-H_h(i,j),Vi,Wj);//wj = wj-Hij*Vi 
        }
        auto Hj_h = Kokkos::subview(H_h,Kokkos::make_pair(0,j+1) ,j);
      }
      else if( ortho == "CGS2"){
        auto V0j = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,j+1)); 
        auto Hj = Kokkos::subview(H,Kokkos::make_pair(0,j+1) ,j);
        auto Hj_h = Kokkos::subview(H_h,Kokkos::make_pair(0,j+1) ,j);
        KokkosBlas::gemv("C", one, V0j, Wj, zero, Hj); // Hj = Vj^T * wj
        KokkosBlas::gemv("N", -one, V0j, Hj, one, Wj); // wj = wj - Vj * Hj

        //Re-orthog CGS:
        ViewVectorType tmp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "tmp"),j+1); 
        KokkosBlas::gemv("C", one, V0j, Wj, zero, tmp); // tmp (Hj) = Vj^T * wj
        KokkosBlas::gemv("N", -one, V0j, tmp, one, Wj); // wj = wj - Vj * tmp 
        KokkosBlas::axpy(one, tmp, Hj); // Hj = Hj + tmp
        Kokkos::deep_copy(Hj_h,Hj);
      }
      else {
        throw std::invalid_argument("Invalid argument for 'ortho'.  Please use 'CGS2' or 'MGS'.");
      }

      MT tmpNrm = KokkosBlas::nrm2(Wj);
      H_h(j+1,j) = tmpNrm; 
      if(tmpNrm < 1e-14){ 
        throw std::runtime_error("GMRES lucky breakdown. Solver terminated without convergence."); 
      }

      Vj = Kokkos::subview(V,Kokkos::ALL,j+1); 
      KokkosBlas::scal(Vj,one/H_h(j+1,j),Wj); // Wj = Vj/H(j+1,j)

      // Givens for real and complex (See Alg 3 in "On computing Givens rotations reliably and efficiently"
      // by Demmel, et. al. 2001)
      // Apply Givens rotation and compute shortcut residual:
      for(int i=0; i<j; i++){
        ST tempVal = CosVal_h(i)*H_h(i,j) + SinVal_h(i)*H_h(i+1,j);
        H_h(i+1,j) = -AT::conj(SinVal_h(i))*H_h(i,j) + CosVal_h(i)*H_h(i+1,j);
        H_h(i,j) = tempVal;
      }
      ST f = H_h(j,j);
      ST g = H_h(j+1,j);
      MT f2 = AT::real(f)*AT::real(f) + AT::imag(f)*AT::imag(f); 
      MT g2 = AT::real(g)*AT::real(g) + AT::imag(g)*AT::imag(g);
      ST fg2 = f2 + g2;
      ST D1 = one / AT::sqrt(f2*fg2); 
      CosVal_h(j) = f2*D1;
      fg2 = fg2 * D1;
      H_h(j,j) = f*fg2;
      SinVal_h(j) = f*D1*AT::conj(g);
      H_h(j+1,j) = zero; 

      GVec_h(j+1) = GVec_h(j)*(-AT::conj(SinVal_h(j)));
      GVec_h(j) = GVec_h(j)*CosVal_h(j);
      shortRelRes = abs(GVec_h(j+1))/nrmB;

      std::cout << "Shortcut relative residual for iteration " << j+(cycle*m) << " is: " << shortRelRes << std::endl;

      //If short residual converged, or time to restart, check true residual
      if( shortRelRes < tol || j == m-1 ) {
        //Compute least squares soln with Givens rotation:
        auto GLsSolnSub_h = Kokkos::subview(GLsSoln_h,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
        auto GVecSub_h = Kokkos::subview(GVec_h, Kokkos::make_pair(0,m));
        Kokkos::deep_copy(GLsSolnSub_h, GVecSub_h); //Copy LS rhs vec for triangle solve.
        auto GLsSolnSub2_h = Kokkos::subview(GLsSoln_h,Kokkos::make_pair(0,j+1),Kokkos::ALL);
        auto H_Sub_h = Kokkos::subview(H_h, Kokkos::make_pair(0,j+1), Kokkos::make_pair(0,j+1)); 
        KokkosBlas::trsm("L", "U", "N", "N", one, H_Sub_h, GLsSolnSub2_h); //GLsSoln = H\GLsSoln
        Kokkos::deep_copy(GLsSoln, GLsSoln_h);

        //Update solution and compute residual with Givens:
        VSub = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,j+1)); 
        Kokkos::deep_copy(Xiter,X); //Can't overwrite X with intermediate solution.
        auto GLsSolnSub3 = Kokkos::subview(GLsSoln,Kokkos::make_pair(0,j+1),0);
        KokkosBlas::gemv ("N", one, VSub, GLsSolnSub3, one, Xiter); //x_iter = x + V(1:j+1)*lsSoln
        KokkosSparse::spmv("N", one, A, Xiter, zero, Wj); // wj = Ax
        Kokkos::deep_copy(Res,B); // Reset r=b.
        KokkosBlas::axpy(-one, Wj, Res); // r = b-Ax. 
        trueRes = KokkosBlas::nrm2(Res);
        relRes = trueRes/nrmB;
        std::cout << "True relative residual for iteration " << j+(cycle*m) << " is : " << relRes << std::endl;
        numIters = j;

        if(relRes < tol){
          converged = true;
          Kokkos::deep_copy(X, Xiter); //Final solution is the iteration solution.
          break; //End Arnoldi iteration. 
        }
      }

    }//end Arnoldi iter.

    cycle++;

    //This is the end, or it's time to restart. Update solution to most recent vector.
    Kokkos::deep_copy(X, Xiter);
  }

  std::cout << "Ending relative residual is: " << relRes << std::endl;
  myStats.endRelRes = relRes;
  if( converged ){
    std::cout << "Solver converged! " << std::endl;
    myStats.convFlagVal = GmresStats::FLAG::Conv;
  }
  else if( shortRelRes < tol ){
    std::cout << "Shortcut residual converged, but solver experienced a loss of accuracy." << std::endl;
    myStats.convFlagVal = GmresStats::FLAG::LOA;
  }
  else{
    std::cout << "Solver did not converge. :( " << std::endl;
    myStats.convFlagVal = GmresStats::FLAG::NoConv;
  }
  myStats.numIters = (cycle-1)*m + numIters;
  std::cout << "The solver completed " << myStats.numIters << " iterations." << std::endl;

  return myStats;
}

