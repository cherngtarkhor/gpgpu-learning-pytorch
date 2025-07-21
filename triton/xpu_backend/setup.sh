#!/bin/bash

# To give this script sudo permission
## sudo visudo
### username ALL=(ALL) NOPASSWD: /path/to/your/setup.sh
#### chmod +x setup.sh

### Step 0: Install Intel Client GPU Drivers ###
## Windows: Install Arc Control (optional) ##
# https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html 

## Ubuntu: Install Client GPU
# https://dgpu-docs.intel.com/driver/client/overview.html#selecting-the-right-operating-system-version

# #Step 0: Setup Proxy
# echo "Configuring Proxy..."
# echo 'export HTTP_PROXY=http://proxy-dmz.intel.com:912' >> ~/.bashrc 
# echo 'export HTTPS_PROXY=http://proxy-dmz.intel.com:912' >> ~/.bashrc 
# echo 'export http_proxy=http://proxy-dmz.intel.com:912' >> ~/.bashrc 
# echo 'export https_proxy=http://proxy-dmz.intel.com:912' >> ~/.bashrc 
# source ~/.bashrc 

# sudo touch /etc/apt/apt.conf.d/proxy.conf  
# echo "Acquire::http::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf 
# echo "Acquire::https::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf 
# echo "Acquire::ftp::Proxy \"http://proxy-dmz.intel.com:912\";" | sudo tee -a /etc/apt/apt.conf.d/proxy.conf 
# echo "Proxy Configuration Complete"

### Step 1: Setup General Dependencies
sudo apt-get update
sudo apt install -y python3-pip
sudo apt install -y python3-venv
sudo apt-get install -y jq
sudo apt install -y gawk #For gawk

### Step 2: Set Up oneAPI Environment Variables ###
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d12ef2ba-7efd-4866-8f85-78eaf40b2fe2/intel-deep-learning-essentials-2025.0.1.27_offline.sh
sudo sh ./intel-deep-learning-essentials-2025.0.1.27_offline.sh -a --silent --eula accept

sudo apt update
sudo apt -y install cmake pkg-config build-essential

echo 'source /opt/intel/oneapi/2025.0/oneapi-vars.sh' >> ~/.bashrc
source ~/.bashrc

rm ./intel-deep-learning-essentials-2025.0.1.27_offline.sh
echo "Environment setup complete!"
