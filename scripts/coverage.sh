#!/bin/bash -el
# This script parses all the generated gcda files and then
# attempts to upload the coverage reports to codecov.
# codecov handles merging of reports across all our
# builds. See https://docs.codecov.com/docs/merging-reports.
if [ ! -e ./CMakeCache.txt ]; then
  echo "ERROR: This script must be run from the cmake build directory."
fi

BUILD_DIR=$PWD
UPLOAD_RETRIES=30
UPLOAD_COUNTER=0
mkdir -p Testing/CoverageInfo
date > Testing/CoverageInfo/gcov.log
pushd Testing/CoverageInfo

# Run gcov on the generated gcda files
for dir in $(cat ${BUILD_DIR}/CMakeFiles/TargetDirectories.txt); do
  if [ -d $dir ]; then
    gcda_files=$(find $dir -iname '*.gcda')
  fi
  if [ ! -z $(echo $gcda_files | head -c 1) ]; then
    echo "STATUS: processing $dir..." | tee -a gcov.log
    gcov -a -m -x $gcda_files 2>&1 >> gcov.log 
    rm -f $gcda_files
    UPLOAD_COUNTER=$UPLOAD_RETRIES
  fi
done

# Upload gcov results to codecov
while [ $UPLOAD_COUNTER -ne 0 ]; do
  echo "STATUS: Uploading to codecov attempt $[$UPLOAD_RETRIES-$UPLOAD_COUNTER]..."
  codecov -X gcovcodecov -X gcov -t $CODECOV_TOKEN --root $BUILD_DIR --name $JOB_NAME --url TODO-KK-CODECOV-URL
  ret=$?
  if [ $ret -eq 0]; then
    exit 0
  fi
  sleep 5s
  UPLOAD_COUNTER=$[$UPLOAD_COUNTER-1]
done