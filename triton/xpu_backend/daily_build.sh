#!/bin/bash

# if [ "$#" -ne 1 ]; then
#     echo "!!!Error!!!"
#     echo "jd@192.198.146.172:/home/jd/wheels/"
#     exit 1
# fi

# USER_NAME=$1
cd ~

git clone https://github.com/intel/intel-xpu-backend-for-triton.git
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: git clone failed with exit code $TEST_EXIT_CODE"
    exit 1
fi

COMMIT=$(git -C ~/intel-xpu-backend-for-triton rev-parse HEAD)
DIRECTORY="${HOME}/${COMMIT:0:7}"
TEST_DIR="${DIRECTORY}/intel-xpu-backend-for-triton"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "TEST_DIR=${TEST_DIR}" >> $GITHUB_ENV
echo "TIMESTAMP=${TIMESTAMP}" >> $GITHUB_ENV

if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY" && mv "${HOME}/intel-xpu-backend-for-triton" "$DIRECTORY"
    TEST_DIR="${DIRECTORY}/intel-xpu-backend-for-triton"
    cd $TEST_DIR
else 
    cd "Commit ID Exists..."
    exit 1
fi

python3 -m venv $TEST_DIR/.devbuild

source $TEST_DIR/.devbuild/bin/activate

echo "Virtual Environment Set"
export MAX_JOBS=8
pip install wheel pybind11 setuptools ninja cmake

echo "Building Pytorch..."
$TEST_DIR/scripts/install-pytorch.sh --source
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Build Pytorch Successfully"
    export MAX_JOBS=8
    echo "Building Triton..."
    $TEST_DIR/scripts/compile-triton.sh 
else
    echo "Build Fail, exit with 1."
    exit 1
fi
# echo "Building Pytorch..."
# $TEST_DIR/scripts/install-pytorch.sh --source
# echo "Building Triton..."
# $TEST_DIR/scripts/compile-triton.sh 

echo "Installing Triton Wheels"
cd python 

echo "Building Triton Wheels..."
DEBUG=1 python setup.py bdist_wheel

mkdir ${TEST_DIR}/wheels
cp -L ${TEST_DIR}/.scripts_cache/pytorch/dist/*.whl ${TEST_DIR}/wheels
cp -L ${TEST_DIR}/python/dist/*.whl ${TEST_DIR}/wheels
ls -lh ${TEST_DIR}/wheels

echo "All Jobs Completed!"
echo "PyTorch is located at ~/$(whoami)/${COMMIT:0:7}/intel-xpu-backend-for-triton/.scripts_cache/pytorch/dist/*.whl"
echo "Triton is located at ~/$(whoami)/${COMMIT:0:7}/intel-xpu-backend-for-triton/python/dist/*.whl"

# Triton Location: intel-xpu-backend-for-triton/python/dist
#scp /home/$COMMITID/intel-xpu-backend-for-triton/python/dist/*.whl $DES_DIR
 
# Pytorch Location: intel-xpu-backend-for-triton/.scripts_cache/pytorch/dist 
#scp /home/$COMMITID/intel-xpu-backend-for-triton/.scripts_cache/pytorch/dist/*.whl $DES_DIR
