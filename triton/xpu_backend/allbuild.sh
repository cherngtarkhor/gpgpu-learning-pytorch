#!/bin/bash

prebuild_setup(){

    # Setup Prebuild Environment
    python3 -m venv $TEST_DIR/.prebuild
    source $TEST_DIR/.prebuild/bin/activate
    echo "Prebuild Environment Set"
    export MAX_JOBS=16

    echo "Building Triton..."
    $TEST_DIR/scripts/compile-triton.sh
    pip uninstall -y triton

    # Install Packages from Nightly Wheels
    echo "**** Download nightly builds. ****"
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

    echo "**** Install PyTorch and pinned dependencies from nightly builds. ****"
    pip install $TEST_DIR/$WHEEL_PATTERN/*.whl
    pip install mkl

    deactivate
    echo "Prebuild Setup Complete..."
}

devbuild_setup(){

    # Setup Devbuild Environment
    python3 -m venv $TEST_DIR/.devbuild
    source $TEST_DIR/.devbuild/bin/activate
    echo "Devbuild Environment Set"
    pip install wheel pybind11 setuptools ninja cmake

    echo "Building Pytorch..."
    export MAX_JOBS=16
    $TEST_DIR/scripts/install-pytorch.sh --source
    TEST_EXIT_CODE=$?
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "Build Pytorch Successfully"
        echo "Building Triton..."
        $TEST_DIR/scripts/compile-triton.sh 
    else
        echo "Build Fail, exit with 1."
        exit 1
    fi

    deactivate
    echo "Devbuild Setup Complete..."
}

# Clone Repository
cd ~

if [ -d "./intel-xpu-backend-for-triton" ]; then 
    sudo rm -r ./intel-xpu-backend-for-triton
fi

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

else 
    echo "Commit ID Exists..."
    sudo rm -r ./intel-xpu-backend-for-triton
fi

# Due the reason that Tutorial 1 Plot and Subprocess will cause the whole test to fail, we disable it.
## Disable Tutorial 1 Plot
gawk -i inplace '{gsub("show_plots=True","show_plots=False"); print $0;}' $DIRECTORY/intel-xpu-backend-for-triton/python/tutorials/01-vector-add.py
## Disable Subprocess 
sed -i '/TRITON_DISABLE_LINE_INFO=1 TRITON_TEST_SUITE=subprocess/,+1 s/^/# /' $TEST_DIR/scripts/test-triton.sh

# Parse flags and arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prebuild)
        prebuild_setup
        shift
        ;;
        --devbuild)
        devbuild_setup
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done







