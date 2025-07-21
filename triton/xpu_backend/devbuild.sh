#!/bin/bash

# Setup repo on run machine
export MAX_JOBS=2
cd ~

## Accept user input for numbers to repeat the test
echo "$#"
if [ "$#" -ne 1 ]; then
    echo "!!!Error!!!"
    echo "Please state your test environment"
    exit 1
fi

TEST_DIR=$1
# Setup Devbuild Environment

# Setup Triton Test
python3 -m venv $TEST_DIR/.devbuild
source $TEST_DIR/.devbuild/bin/activate
echo "Building Triton..."
$TEST_DIR/scripts/compile-triton.sh
pip uninstall -y triton

# On Run Machine
# Install Packages from Nightly Wheels

echo "**** Download DevBuilds... ****"
PYTHON_VERSION=3.12
echo "Python Version: $PYTHON_VERSION"

RUN_ID=$(gh run list -w "Daily_Build" -R intel-sandbox/personal.flexpg.tritonxpu-common --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
echo "Run ID:$RUN_ID"

#TEMP_DIR=$(mktemp -d)
WHEEL_PATTERN="wheels-devbuild-${PYTHON_VERSION}*"
echo "Wheel Pattern: $WHEEL_PATTERN"

echo "**** Downloading dependencies for DevBuilds... ****"
gh run download $RUN_ID \
    --repo intel-sandbox/personal.flexpg.tritonxpu-common \
    --pattern "$WHEEL_PATTERN" \
    --dir $TEST_DIR
    #--dir $TEMP_DIR
#cd $TEMP_DIR/$WHEEL_PATTERN
echo "**** Installing DevBuild... ****"
pip install $TEST_DIR/$WHEEL_PATTERN/*.whl
echo "Devbuild Setup Complete..."
#rm -rf $TEMP_DIR

