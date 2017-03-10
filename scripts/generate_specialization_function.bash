#!/bin/bash

Function=$1
MasterHeader=$2
ScalarList="double float Kokkos::complex<double> Kokkos::complex<float>"
LayoutList="LayoutLeft LayoutRight"
ExecMemSpaceList="Cuda,CudaSpace OpenMP,HostSpace Pthread,HostSpace Serial,HostSpace"


filename_hpp=${Function}_decl_specialization.hpp
cat ../../scripts/header > generated_specializations/${filename_hpp}

for Scalar in ${ScalarList}; do
for Layout in ${LayoutList}; do
for ExecMemSpace in ${ExecMemSpaceList}; do
   ExecMemSpaceArray=(${ExecMemSpace//,/ })
   ExecSpace=${ExecMemSpaceArray[0]}
   MemSpace=${ExecMemSpaceArray[1]}
   echo "Generate: " ${Function} " " ${Scalar} " " ${Layout} " " ${ExecSpace} " " ${MemSpace}
   ../../scripts/generate_specialization_type.bash ${Function} ${Scalar} ${Layout} ${ExecSpace} ${MemSpace} ${MasterHeader}
done
done
done
