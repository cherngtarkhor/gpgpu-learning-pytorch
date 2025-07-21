#!/bin/bash
#to give this script sudo permission
#sudo visudo
#username ALL=(ALL) NOPASSWD: /path/to/your/setup.sh
# chmod +x setup.sh

### Step 1: Install Intel Client GPU Drivers ###
## Install Arc Control (optional) ##
# https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html 

echo "Installing Intel Client GPU Drivers..."
echo " "
sudo apt-get update
sudo apt install -y python3-pip
sudo apt install -y python3-venv
sudo apt-get install -y jq
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list
sudo apt update
sudo apt-get install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo libze-dev intel-ocloc

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
sudo apt install intel-for-pytorch-gpu-dev-0.5 intel-pti-dev-0.9

### Step 4: Set Up oneAPI Environment Variables ###
sed -i -e '$ a source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh' \
       -e '$ a source /opt/intel/oneapi/pti/0.9/env/vars.sh' ~/.bashrc

source ~/.bashrc

echo "Environment setup complete!"
