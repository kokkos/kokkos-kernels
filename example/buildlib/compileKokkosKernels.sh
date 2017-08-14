KOKKOS_PATH=${HOME}/work/kokkos #path to kokkos source
KOKKOSKERNELS_SCALARS=double #the scalar types to instantiate =double,float...
KOKKOSKERNELS_LAYOUTS=LayoutLeft #the layout types to instantiate.
KOKKOSKERNELS_ORDINALS=int,long #ordinal types to instantiate
KOKKOSKERNELS_OFFSETS=int,size_t #offset types to instantiate
KOKKOSKERNELS_PATH=../.. #path to kokkos-kernels top directory.
CXX=${KOKKOS_PATH}/config/nvcc_wrapper #icpc #
KOKKOSKERNELS_OPTIONS=eti-only #options for kokkoskernels  
KOKKOS_DEVICES=Cuda,OpenMP #devices Cuda...
KOKKOS_ARCHS=Pascal60,Power8


../../scripts/generate_makefile.bash --kokkoskernels-path=${KOKKOSKERNELS_PATH} --with-scalars=${KOKKOSKERNELS_SCALARS} --with-ordinals=${KOKKOSKERNELS_ORDINALS} --with-offsets=${KOKKOSKERNELS_OFFSETS} --kokkos-path=${KOKKOS_PATH} --with-devices=${KOKKOS_DEVICES} --arch=${KOKKOS_ARCHS} --compiler=${CXX} --with-options=${KOKKOSKERNELS_OPTIONS}
