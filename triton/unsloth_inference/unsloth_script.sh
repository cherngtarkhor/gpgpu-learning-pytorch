#!/bin/bash

echo "Currently at $(pwd)"

if [ -e "./unsloth_inference/outputs" ]; then 
    echo "Output Directory Exists..."
else
    echo "Creating output directory..."
    mkdir ./unsloth_inference/outputs
    echo "Output directory created..."
fi

cd ./unsloth_inference
echo "Currently at $(pwd)"
sleep 10

if [ ! -d "./unsloth" ]; then
    git clone https://github.com/cherngtar-intel/unsloth.git -b unsloth_xpu_20241210 
    if [ $? -ne 0 ]; then
        echo "Failed to clone the repository."
        exit 1
    fi
    echo "Unsloth Directory Cloned"
else
    echo "Unsloth Directory Exists..."
fi

cd ./unsloth
echo "Currently at $(pwd)"

echo "Updating Submodules..."
git submodule update --init --recursive
if [ $? -ne 0 ]; then
    echo "Failed to update submodules."
    exit 1
fi

echo "Checking Virtual Environment..."
if [ ! -d ".venv" ]; then
    echo "No virtual environment detected. Creating Virtual Environment..."
    python -m venv .venv --prompt unsloth
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual Environment Exists..."
fi

echo "Activating Virtual Environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Checking Triton and Pytorch..."
if pip list | grep -q 'torch' && pip list | grep -q 'triton'; then
    echo "Both torch and triton are installed."
else
    PYTHON_VERSION=3.12
    echo "> Python Version: $PYTHON_VERSION"

    RUN_ID=$(gh run list -w "Triton wheels" -R intel/intel-xpu-backend-for-triton --json databaseId,conclusion | jq -r '[.[] | select(.conclusion=="success")][0].databaseId')
    if [ -z "$RUN_ID" ]; then
        echo "Failed to get the run ID."
        exit 1
    fi
    echo "> Run ID:$RUN_ID"

    WHEEL_PATTERN="wheels-pytorch-py${PYTHON_VERSION}*"
    echo "> Wheel Pattern: $WHEEL_PATTERN"

    echo "**** Downloading dependencies from nightly builds. ****"
    gh run download $RUN_ID \
    --repo intel/intel-xpu-backend-for-triton \
    --pattern "$WHEEL_PATTERN" \
    --dir .
    if [ $? -ne 0 ]; then
        echo "Failed to download dependencies."
        exit 1
    fi
    echo "Finish downloading dependencies..."

    echo "Installing Triton and Pytorch..."
    pip install ./$WHEEL_PATTERN/*torch-*.whl ./$WHEEL_PATTERN/triton-*.whl
    if [ $? -ne 0 ]; then
        echo "Failed to install Triton and Pytorch."
        exit 1
    fi
    echo "Finish installing Triton and Pytorch..."
fi

echo "Installing Dependencies..."
pip install -r requirements.txt
pip install transformers==4.47.0 #Temporary

echo "Running Benchmark..."
python3 -u unsloth_script.py --benchmark all
if [ $? -ne 0 ]; then
    echo "Benchmark script failed."
    exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
NEWEST_FILE=$(ls -t *.csv | head -1)
TARGET_DIR="../outputs"
cp $NEWEST_FILE $TARGET_DIR

cd $TARGET_DIR
echo "Currently at $(pwd)"

CSV_FILE="output_${TIMESTAMP}.csv"
mv "$NEWEST_FILE" "$CSV_FILE"
if [ $? -ne 0 ]; then
    echo "Failed to move the CSV file."
    exit 1
fi

echo "Adding CSV file to the repository..."
git fetch origin
git merge origin/main
git add "$CSV_FILE"
git commit -m "Uploaded ${CSV_FILE}"
git push origin main
if [ $? -ne 0 ]; then
    echo "Failed to push the CSV file to the repository."
    exit 1
fi

echo "Script completed successfully."