#!/bin/bash

# ./testboth.sh > output.txt
# grep ">>>\|>> \|Timer:" output.txt

numacmd="KMP_AFFINITY=balanced numactl --membind 1"
sz="-ni 32 -nj 32 -nk 128"

for bsz in 4 5 8 9 15 16; do
    echo ">>> bsz $bsz"
    for nth in 4 8 16 32 34 64 68 128 136 256 272; do
        echo ">> nthread $nth"
        echo "> kk"
        cmd="$numacmd ./KokkosKernels_Test_BlockCrs --kokkos-threads=$nth $sz -bs $bsz"
        echo $cmd
        eval $cmd
        echo "> sparc"
        cmd="OMP_NUM_THREADS=$nth $numacmd ../../ref/a.out $sz -bs $bsz"
        echo $cmd
        eval $cmd
    done
done
