#!/bin/bash

export MAX_JOBS=2
cd ~

git clone https://github.com/intel/intel-xpu-backend-for-triton.git
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: git clone failed with exit code $TEST_EXIT_CODE"
    exit 1
fi

COMMIT=$(git -C ./intel-xpu-backend-for-triton rev-parse HEAD)
DIRECTORY="${HOME}/${COMMIT:0:7}"
TEST_DIR="${DIRECTORY}/intel-xpu-backend-for-triton"

echo "test_dir=${TEST_DIR}" >> $GITHUB_OUTPUT

if [ ! -d "$DIRECTORY" ]; then
    mkdir "$DIRECTORY" && mv "${HOME}/intel-xpu-backend-for-triton" "$DIRECTORY"
    cd "$TEST_DIR"
else 
    cd "Commit ID Exists..."
    exit 1
fi

# Disable Tutorial 1 Plot
gawk -i inplace '{gsub("show_plots=True","show_plots=False"); print $0;}' $DIRECTORY/intel-xpu-backend-for-triton/python/tutorials/01-vector-add.py
# Disable Subprocess 
sed -i '/TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess/,+1 s/^/# /' $TEST_DIR/scripts/test-triton.sh

# Setup Prebuild Environment
# Setup Triton Test
python3 -m venv $TEST_DIR/.prebuild
source $TEST_DIR/.prebuild/bin/activate
echo "Building Triton..."
$TEST_DIR/scripts/compile-triton.sh
pip uninstall -y triton

# Install Packages from Nightly Wheels
echo "**** Download nightly builds. ****"
sudo apt-get install -y jq 
PYTHON_VERSION=3.12
echo "Python Version: $PYTHON_VERSION"

RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
echo "Run ID:$RUN_ID"

#TEMP_DIR=$(mktemp -d)
WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
echo "Wheel Pattern: $WHEEL_PATTERN"

echo "**** Downloading dependencies from nightly builds. ****"
gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "$WHEEL_PATTERN" \
    --dir $TEST_DIR
    #    --dir $TEMP_DIR
#cd $TEMP_DIR/$WHEEL_PATTERN
#cd ./$WHEEL_PATTERN  
echo "**** Install PyTorch and pinned dependencies from nightly builds. ****"
pip install $TEST_DIR/$WHEEL_PATTERN/*.whl
pip install mkl
echo "Prebuild Setup Complete..."
#rm -rf $TEMP_DIR
