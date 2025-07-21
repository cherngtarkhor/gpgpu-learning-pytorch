#!/bin/bash
#to give this script sudo permission
#sudo visudo
#username ALL=(ALL) NOPASSWD: /path/to/your/setup.sh
# chmod +x setup.sh

#to run this script with log
# setup.sh |& tee -a setup_log.txt 

## Accept user input compile.sh dir ##
if [ "$#" -ne 1 ]; then
    echo "Command to call this script: $0 <venv name>"
    echo "<venv name> example: ".devbuild""
    exit 1
fi

VENV_NAME=$1

### Step 1: Install Intel Client GPU Drivers ###
## Install Arc Control (optional) ##
# https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html 

echo "Installing Intel Client GPU Drivers..."
echo " "
sudo apt-get update
sudo apt install -y python3-pip 
sudo apt install -y python3-venv 
sudo apt-get install -y jq
sudo apt-get install -y intel-ocloc intel-opencl-icd libze1 libze-intel-gpu1

### Step 2: Install Intel Level Zero ###
sudo apt-get install -y cmake g++
git config --global http.proxy http://proxy-dmz.intel.com:912 
git config --global https.proxy http://proxy-dmz.intel.com:912 
git clone https://github.com/oneapi-src/level-zero.git 
cd level-zero 
mkdir build && cd build 
cmake .. -D CMAKE_BUILD_TYPE=Debug 
cmake --build . --target package 
sudo cmake --build . --target install 
cd ~

### Step 3: Install Intel Support Packages ###

echo "Installing Intel Support Packages..."
echo " "

# sed -i -e '$ a export HTTP_PROXY=http://proxy-dmz.intel.com:912' \
#        -e '$ a export HTTPS_PROXY=http://proxy-dmz.intel.com:912' \
#        -e '$ a export http_proxy=http://proxy-dmz.intel.com:912' \
#        -e '$ a export https_proxy=http://proxy-dmz.intel.com:912' ~/.bashrc
# source ~/.bashrc

wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor > /tmp/intel-for-pytorch-gpu-dev-keyring.gpg 
sudo mv /tmp/intel-for-pytorch-gpu-dev-keyring.gpg /usr/share/keyrings
echo "deb [signed-by=/usr/share/keyrings/intel-for-pytorch-gpu-dev-keyring.gpg] https://apt.repos.intel.com/intel-for-pytorch-gpu-dev all main" > /tmp/intel-for-pytorch-gpu-dev.list
sudo mv /tmp/intel-for-pytorch-gpu-dev.list /etc/apt/sources.list.d 

# sudo touch /etc/apt/apt.conf.d/proxy.conf 
# echo "Acquire::http::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf
# echo "Acquire::https::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf
# echo "Acquire::ftp::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf

sudo apt update  
sudo apt install -y intel-for-pytorch-gpu-dev-0.5 intel-pti-dev

### Step 4: Set Up oneAPI Environment Variables ###
sed -i -e '$ a source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh' \
       -e '$ a source /opt/intel/oneapi/pti/latest/env/vars.sh' ~/.bashrc

source ~/.bashrc

echo "Environment setup complete!"

# ## Step 1: Clone Git Repository and Create Virtual Environment ###
# will clone sudo mv /tmp/intel-for-pytorch-gpu-dev.list /etc/apt/sources.list.dto /opt dircd opt
cd ~
git clone https://github.com/intel/intel-xpu-backend-for-triton.git 
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "Error: git clone failed with exit code $TEST_EXIT_CODE"
    exit 1
fi

COMMIT=$(git -C ~/intel-xpu-backend-for-triton rev-parse HEAD)

mkdir ~/${COMMIT:0:7} && mv ~/intel-xpu-backend-for-triton ~/${COMMIT:0:7}

cd ~/${COMMIT:0:7}/intel-xpu-backend-for-triton 

#git reset --hard 0a8fedfb244fb3413fb64402e617d7755f79fc81
export MAX_JOBS=2

python3 -m venv $VENV_NAME

source $VENV_NAME/bin/activate

echo "Building Pytorch..."
scripts/install-pytorch.sh --source
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "Build Pytorch Successfully"
    export MAX_JOBS=2
    echo "Building Triton..."
    scripts/compile-triton.sh 
else
    echo "Build Fail, exit with 0."
fi





