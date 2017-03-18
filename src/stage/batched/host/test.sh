exec=$1
numacmd="numactl --membind 1"
testfile=$exec.txt
rm -f $testfile

echo $exec > $testfile

for th in 1 2 4 8 16 32 34 64 68 136 272; do
    echo "$numacmd ./$exec --kokkos-threads=$th  >> $testfile"
    $numacmd ./$exec --kokkos-threads=$th  >> $testfile
done
