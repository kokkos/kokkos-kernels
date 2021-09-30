#!/bin/bash
################################################################################
# @Brief: On the specified arch, build and run KokkosBlas3_perf_test.
#
# The value of this script is to ensure that the benchmark results can be easily
# reproduced.
#
# Author: Evan Harvey <eharvey@sandia.gov>
################################################################################

function envprint() {
  for x in $@; do
      echo $x:\$$x | envsubst
  done
}

function printhelp() {
  echo "--Usage--"
  echo "$0 PRECISION HOST_ARCH <ACCELERATOR_ARCH>"
  echo "  PRECISION:        Kokkos::Experimental::half_t, float, double"
  echo "  HOST_ARCH:        POWER9, A64FX, SKX"
  echo "  ACCELERATOR_ARCH: VOLTA70"
  echo ""
}

function earlyexit() {
   rm -rf $benchmark_dir
   exit $1
}

function beval() {
  local ret=0
  echo "---------------------------------------------------------------------------------------------------------------"
  echo "START: \"$@\""
  if [ $dry_run == "off" ]; then
    eval $@
    ret=$PIPESTATUS
  fi
  if [ $ret -ne 0 ]; then
      echo "ERROR: \"$@\""
      earlyexit 1
  fi
  echo "END  : \"$@\""
  echo "---------------------------------------------------------------------------------------------------------------"
}

# Handle input args
export KOKKOS_SRC_DIR=${KOKKOS_SRC_DIR:-"$HOME/KOKKOS.base/kokkos"}
export KOKKOS_SRC_DIR=$(realpath $KOKKOS_SRC_DIR)
export KOKKOS_SHA=${KOKKOS_SHA:-"2fc1050"} # Tip of develop as of 09-30-21
export KOKKOSKERNELS_SRC_DIR=${KOKKOSKERNELS_SRC_DIR:-"$HOME/KOKKOS.base/kokkos-kernels"}
export KOKKOSKERNELS_SRC_DIR=$(realpath $KOKKOSKERNELS_SRC_DIR)
export KOKKOSKERNELS_SHA=${KOKKOSKERNELS_SHA:-"3d2992f"} # Tip of e10harvey/issue1045 as of 09-30-21
envprint KOKKOS_SRC_DIR KOKKOS_SHA KOKKOSKERNELS_SRC_DIR KOKKOSKERNELS_SHA

# Create benchmark directory
benchmark_dir=$0_$(date +"%Y-%m-%d_%H.%M.%S")
mkdir -p $benchmark_dir/kokkos-{build,instal}
mkdir -p $benchmark_dir/kokkos-kernels-{build,install}
export KOKKOS_BUILD_DIR=$(realpath $benchmark_dir/kokkos-build)
export KOKKOS_INSTALL_DIR=$(realpath $benchmark_dir/kokkos-install)
export KOKKOSKERNELS_BUILD_DIR=$(realpath $benchmark_dir/kokkos-kernels-build)
export KOKKOSKERNELS_INSTALL_DIR=$(realpath $benchmark_dir/kokkos-kernels-install)
envprint KOKKOS_INSTALL_DIR KOKKOS_BUILD_DIR KOKKOSKERNELS_BUILD_DIR KOKKOSKERNELS_INSTALL_DIR

dry_run="off"
precision="$1"
arch_names="$2 $3"
echo "PRECISION=\"$1\", HOST_ARCH=\"$2\", ACCELERATOR_ARCH=\"$3\""

# Setup arch specific cmake configurations and job submission commands
if [[ "$arch_names" == " " || -z $precision ]]; then
    printhelp; earlyexit 1
elif [ "$arch_names" == "POWER9 VOLTA70" ]; then
  module load cmake/3.18.0 gcc/7.2.0 cuda/10.2.2
  kokkos_config_cmd="cd $KOKKOS_BUILD_DIR; $KOKKOS_SRC_DIR/generate_makefile.bash --cxxflags='-O3' --arch=Power9,Volta70 \
                     --with-cuda=$CUDA_PATH --compiler=$KOKKOS_SRC_DIR/bin/nvcc_wrapper --kokkos-path=$KOKKOS_SRC_DIR \
                     --prefix=$KOKKOS_INSTALL_DIR 2>&1 | tee kokkos_config_cmd.out"

  kokkoskernels_config_cmd="cd $KOKKOSKERNELS_BUILD_DIR; $KOKKOSKERNELS_SRC_DIR/cm_generate_makefile.bash \
                            --cxxflags='-O3' --arch=Power9,Volta70 \
                            --with-scalars="$precision" \
                            --with-cuda=$CUDA_PATH --compiler=$KOKKOS_INSTALL_DIR/bin/nvcc_wrapper \
                            --kokkos-path=$KOKKOS_SRC_DIR --kokkoskernels-path=$KOKKOSKERNELS_SRC_DIR \
                            --kokkos-prefix=$KOKKOS_INSTALL_DIR --prefix=$KOKKOSKERNELS_INSTALL_DIR 2>&1 | \
                            tee kokkoskernels_config_cmd.out"
  kokkoskernels_config_layout_cmd="cd $KOKKOSKERNELS_BUILD_DIR; cmake -DKokkosKernels_INST_LAYOUTLEFT:BOOL=OFF \
                                   -DKokkosKernels_INST_LAYOUTRIGHT:BOOL=ON \
                                   $KOKKOSKERNELS_SRC_DIR 2>&1 | tee -a kokkoskernels_config_cmd.out"

  kokkos_build_cmd="bsub -q rhel7W -W 2:00 -Is $KOKKOS_BUILD_DIR/build.sh"
  kokkoskernels_build_cmd="bsub -q rhel7W -W 2:00 -Is $KOKKOSKERNELS_BUILD_DIR/build.sh"
  benchmark_cmd="bsub -q rhel7W -W 2:00 -Is $KOKKOSKERNELS_BUILD_DIR/bench.sh"
elif [ "$arch_names" == "A64FX " ]; then
  earlyexit 0
elif [ "$arch_names" == "SKX " ]; then
  earlyexit 0
else
  echo "Invalid arch: $arch_names"
  printhelp; earlyexit 1
fi

# Set the arch agnostic commands
echo "#!/bin/bash" > $KOKKOS_BUILD_DIR/build.sh
echo "cd $KOKKOS_BUILD_DIR" >> $KOKKOS_BUILD_DIR/build.sh
echo "make -j40 install" >> $KOKKOS_BUILD_DIR/build.sh
chmod +x $KOKKOS_BUILD_DIR/build.sh

echo "#!/bin/bash" > $KOKKOSKERNELS_BUILD_DIR/build.sh
echo "cd $KOKKOSKERNELS_BUILD_DIR/perf_test/blas/blas3" >> $KOKKOSKERNELS_BUILD_DIR/build.sh
echo "make -j40" >> $KOKKOSKERNELS_BUILD_DIR/build.sh
chmod +x $KOKKOSKERNELS_BUILD_DIR/build.sh

echo "#!/bin/bash" > $KOKKOSKERNELS_BUILD_DIR/bench.sh
echo "cd $benchmark_dir" >> $KOKKOSKERNELS_BUILD_DIR/bench.sh
echo "Writing output to: $benchmark_dir/bench.csv..." >> $KOKKOSKERNELS_BUILD_DIR/bench.sh
echo "KOKKOSKERNELS_BUILD_DIR/perf_test/blas/blas3/KokkosBlas3_perf_test \
      --precision=$precision \
      --test=batched_heuristic --routines=gemm --loop_type=parallel --batch_size_last_dim=0 \
      --matrix_size_start=2x2,2x2,2x2 --matrix_size_stop=64x64,64x64,64x64
      --matrix_size_step=2 --batch_size=$((80*1024)) \
      --warm_up_loop=10 --iter=20 --verify=0 \
      --csv=$benchmark_dir/bench.csv" \
       >> $KOKKOSKERNELS_BUILD_DIR/bench.sh
chmod +x $KOKKOSKERNELS_BUILD_DIR/bench.sh

# Check out the correct SHAs
beval "cd $KOKKOS_SRC_DIR && git checkout $KOKKOS_SHA"
beval "cd $KOKKOSKERNELS_SRC_DIR && git checkout $KOKKOSKERNELS_SHA"

# Build Kokkos
beval $kokkos_config_cmd
beval $kokkos_build_cmd

# Build KokkosKernels
beval $kokkoskernels_config_cmd
beval $kokkoskernels_config_layout_cmd
beval $kokkoskernels_build_cmd

# Run the benchmark
beval $benchmark_cmd