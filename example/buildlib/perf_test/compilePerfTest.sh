KOKKOS_PATH=${HOME}/work/kokkos #path to kokkos source
KOKKOSKERNELS_SCALARS=double #the scalar types to instantiate =double,float...
KOKKOSKERNELS_LAYOUTS=LayoutLeft,LayoutRight #the layout types to instantiate.
KOKKOSKERNELS_ORDINALS=int,int64_t #ordinal types to instantiate
KOKKOSKERNELS_OFFSETS=int,size_t #offset types to instantiate
KOKKOSKERNELS_PATH=../../.. #path to kokkos-kernels top directory.
CXX=icpc #
KOKKOSKERNELS_OPTIONS=eti-only #options for kokkoskernels  
KOKKOS_DEVICES=OpenMP,Serial #devices Cuda...

make build -j -f ${KOKKOSKERNELS_PATH}/perf_test/Makefile KOKKOS_PATH=${KOKKOS_PATH} KOKKOSKERNELS_SCALARS=${KOKKOSKERNELS_SCALARS} KOKKOSKERNELS_LAYOUTS=${KOKKOSKERNELS_LAYOUTS} KOKKOSKERNELS_ORDINALS=${KOKKOSKERNELS_ORDINALS} KOKKOSKERNELS_OFFSETS=${KOKKOSKERNELS_OFFSETS}  KOKKOSKERNELS_PATH=${KOKKOSKERNELS_PATH} CXX=${CXX}  KOKKOSKERNELS_OPTIONS=${KOKKOSKERNELS_OPTIONS} KOKKOS_DEVICES=${KOKKOS_DEVICES}

