#!/bin/bash
Function=$1
Scalar=$2
Layout=$3
ExecSpace=$4
MemSpace=$5
filename_master_hpp=$6

Macro=`echo ${Function} | awk '{print toupper($0)}'`
Scalar_UpperCase=`echo ${Scalar} | awk '{print toupper($0)}' | sed 's|\:\:|\_|g' | sed 's|<|_|g' | sed 's|>|_|g'`
Scalar_FileName=`echo ${Scalar} | sed 's|\:\:|\_|g' | sed 's|<|_|g' | sed 's|>|_|g'`
Layout_UpperCase=`echo ${Layout} | awk '{print toupper($0)}'`
ExecSpace_UpperCase=`echo ${ExecSpace} | awk '{print toupper($0)}'`
MemSpace_UpperCase=`echo ${MemSpace} | awk '{print toupper($0)}'`

filename_cpp=generated_specializations/${Function}_inst_specialization_${Scalar_FileName}_${Layout}_${ExecSpace}_${MemSpace}.cpp
filename_hpp=generated_specializations/${Function}_decl_specialization.hpp


cat ../../scripts/header > ${filename_cpp}
echo "" >> ${filename_cpp}
echo "#include \"${filename_master_hpp}\"" >> ${filename_cpp}
echo "" >> ${filename_cpp}
echo "#if defined (KOKKOSKERNELS_INST_SCALAR_${Scalar_UpperCase} \\" >> ${filename_cpp} 
echo " && defined (KOKKOSKERNELS_INST_LAYOUT_${Layout_UpperCase} \\" >> ${filename_cpp} 
echo " && defined (KOKKOSKERNELS_INST_EXECSPACE_${ExecSpace_UpperCase} \\" >> ${filename_cpp} 
echo " && defined (KOKKOSKERNELS_INST_EXECSPACE_${MemSpace_UpperCase} \\" >> ${filename_cpp} 
echo " ${Macro}_DEF(${Scalar}, Kokkos::${Layout}, Kokkos::${ExecSpace}, Kokkos::${MemSpace})" >> ${filename_cpp}
echo "#endif" >> ${filename_cpp}

echo "" >> ${filename_hpp}
echo "#if defined (KOKKOSKERNELS_INST_SCALAR_${Scalar_UpperCase} \\" >> ${filename_hpp}
echo " && defined (KOKKOSKERNELS_INST_LAYOUT_${Layout_UpperCase} \\" >> ${filename_hpp}
echo " && defined (KOKKOSKERNELS_INST_EXECSPACE_${ExecSpace_UpperCase} \\" >> ${filename_hpp}
echo " && defined (KOKKOSKERNELS_INST_EXECSPACE_${MemSpace_UpperCase} \\" >> ${filename_hpp}
echo " ${Macro}_DECL(${Scalar}, Kokkos::${Layout}, Kokkos::${ExecSpace}, Kokkos::${MemSpace})" >> ${filename_hpp}
echo "#endif" >> ${filename_hpp}
