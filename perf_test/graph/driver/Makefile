KOKKOS_PATH = ${HOME}/work/kokkos/
KOKKOS_SRC_PATH = ${KOKKOS_PATH}
SRC = $(wildcard ../*cpp) #../KokkosKernels_SPMM_KK.cpp)
KERNELINC=-I../../../src/stage/graph/impl -I../../../src/stage/graph/ -I../../../src/stage/graph/utils -I../../../src/ -I../../../src/impl/

vpath %.cpp $(sort $(dir $(SRC)))

SPGEMM_EXE = KokkosKernels_SPMM_KK
COLOR_EXE = KokkosKernels_GraphColor
PCG_EXE= KokkosKernels_PCG
BINWRITER_EXE= KokkosKernels_WriteBinaryFromMTX
BINWRITER_EXE2= KokkosKernels_WriteBinaryFromSRCDST


default: $(SPGEMM_EXE) $(COLOR_EXE) $(BINWRITER_EXE) $(BINWRITER_EXE2)

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = icpc
CXXFLAGS = -DGCC_EXTENSION -D_GLIBCXX_PARALLEL -O3 -DKOKKOS_HAVE_HBWSPACE=1 -DKOKKOSKERNELS_HAVE_OUTER
CXXFLAGS = -DGCC_EXTENSION -D_GLIBCXX_PARALLEL -O3 -DKOKKOSKERNELS_HAVE_OUTER

LINK = ${CXX}# -lmemkind
LINKFLAGS =  
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "KNL"
endif



DEPFLAGS = -M

OBJ= KokkosKernels_SPMM_KK.o 
OBJC=KokkosKernels_GraphColoringTest.o
OBJP=KokkosKernels_PCG_GS_TEST.o
OBJW=KokkosKernels_WriteBinaryFromMTX.o
OBJW2=KokkosKernels_WriteBinaryFromBinSrcDst.o

LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

$(SPGEMM_EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(SPGEMM_EXE)
$(COLOR_EXE): $(OBJC) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJC) $(KOKKOS_LIBS) $(LIB) -o $(COLOR_EXE)
$(PCG_EXE): $(OBJP) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJP) $(KOKKOS_LIBS) $(LIB) -o $(PCG_EXE)
$(BINWRITER_EXE): $(OBJW) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJW) $(KOKKOS_LIBS) $(LIB) -o $(BINWRITER_EXE)

$(BINWRITER_EXE2): $(OBJW2) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJW2) $(KOKKOS_LIBS) $(LIB) -o $(BINWRITER_EXE2)
depend:
	makedepend -Y ${SRC} $(KERNELINC)


clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) 
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) $(KERNELINC) -c $< -o $(notdir $@)
# DO NOT DELETE

../KokkosKernels_WriteBinaryFromBinSrcDst.o: ../../../src/stage/graph/utils/KokkosKernels_IOUtils.hpp
../KokkosKernels_WriteBinaryFromBinSrcDst.o: ../../../src/stage/graph/utils/KokkosKernels_SimpleUtils.hpp
../KokkosKernels_WriteBinaryFromBinSrcDst.o: ../../../src/Kokkos_ArithTraits.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_GraphColor.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/impl/KokkosKernels_GraphColor_impl.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_GraphColorHandle.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_Utils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/Kokkos_ArithTraits.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_ExecSpaceUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_SimpleUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_SparseUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_IOUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_PrintUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/utils/KokkosKernels_VectorUtils.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_Handle.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_GaussSeidelHandle.hpp
../KokkosKernels_GraphColoringTest.o: ../../../src/stage/graph/KokkosKernels_SPGEMMHandle.hpp
../KokkosKernels_WriteBinaryFromMTX.o: ../../../src/stage/graph/utils/KokkosKernels_IOUtils.hpp
../KokkosKernels_WriteBinaryFromMTX.o: ../../../src/stage/graph/utils/KokkosKernels_SimpleUtils.hpp
../KokkosKernels_WriteBinaryFromMTX.o: ../../../src/Kokkos_ArithTraits.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_Utils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_ArithTraits.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_ExecSpaceUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_SimpleUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_SparseUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_IOUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_PrintUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/utils/KokkosKernels_VectorUtils.hpp
../KokkosKernels_PCG_GS_TEST.o: ../KokkosKernels_PCG.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_GaussSeidel.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_Handle.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_GraphColorHandle.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_GaussSeidelHandle.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_SPGEMMHandle.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/impl/KokkosKernels_GaussSeidel_impl.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/KokkosKernels_GraphColor.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/stage/graph/impl/KokkosKernels_GraphColor_impl.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Sparse.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Sparse_CrsMatrix.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Sparse_findRelOffset.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Sparse_impl_spmv.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_InnerProductSpaceTraits.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Blas1_MV.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Blas1.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_V_impl_dot.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_abs.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_abs_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_axpby.hpp
../KokkosKernels_PCG_GS_TEST.o: KokkosKernels_config.h
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_V_axpby_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_axpby_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_dot.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_V_dot_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_dot_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_fill.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_fill_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_mult.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_mult_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_nrm1.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_nrm1_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_nrm2.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_nrm2_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_nrm2w.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_nrm2w_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_nrmInf.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_nrmInf_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_recip.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_recip_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_scal.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_scal_multicoeff_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_scal_singlecoeff_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_V_scal_singlecoeff_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_sum.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_sum_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Blas1_MV_impl_update.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_V_update_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosBlas1_impl_MV_update_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Sparse_impl_spmv_omp.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosSparse_impl_MV_spmv_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/generated_specializations_hpp/KokkosSparse_impl_V_spmv_decl_specializations.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Sparse_trsv.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/impl/Kokkos_Sparse_impl_trsm.hpp
../KokkosKernels_PCG_GS_TEST.o: ../../../src/Kokkos_Sparse_MV.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_IOUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_SimpleUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/Kokkos_ArithTraits.hpp
../KokkosKernels_SPMM_KK.o: ../KokkosKernels_RunMultiMemSpGEMM.hpp
../KokkosKernels_SPMM_KK.o: ../KokkosKernels_RunSpGemm.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_SPGEMM.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_Handle.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_GraphColorHandle.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_Utils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_ExecSpaceUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_SparseUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_PrintUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/utils/KokkosKernels_VectorUtils.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_GaussSeidelHandle.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_SPGEMMHandle.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_viennaCL_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_cuSPARSE_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_CUSP_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_mkl_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_mkl2phase_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_HashmapAccumulator.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_Uniform_Initialized_MemoryPool.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/KokkosKernels_GraphColor.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_GraphColor_impl.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_imp_outer.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_memaccess.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_kkmem.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_color.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_speed.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_compression.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_def.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_symbolic.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_triangle.hpp
../KokkosKernels_SPMM_KK.o: ../../../src/stage/graph/impl/KokkosKernels_SPGEMM_impl_seq.hpp
../KokkosKernels_SPMM_KK.o: ../KokkosKernels_MyCRSMatrix.hpp
