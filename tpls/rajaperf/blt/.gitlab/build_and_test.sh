#!/bin/bash

set -e

BUILD_DIR=`pwd`/build

echo "~~~~~~~~~~ START:build_and_test.sh ~~~~~~~~~~~"
echo "CWD="`pwd`
echo "BUILD_DIR="$BUILD_DIR
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

cmake -C ../host-configs/llnl/$SYS_TYPE/$HOST_CONFIG ../tests/internal
make -j8
ctest -DCTEST_OUTPUT_ON_FAILURE=1 --no-compress-output -T Test -VV
xsltproc -o junit.xml ../tests/ctest-to-junit.xsl Testing/*/Test.xml

echo "~~~~~~~~~~ END:build_and_test.sh ~~~~~~~~~~~~~"
echo "CWD="`pwd`
echo "BUILD_DIR="$BUILD_DIR
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
