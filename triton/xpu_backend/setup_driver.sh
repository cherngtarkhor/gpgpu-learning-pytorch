#!/bin/bash

# Setup GPU Driver on Ubuntu
## Source: https://dgpu-docs.intel.com/driver/client/overview.html

# Install the Intel graphics GPG public key
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# Ubuntu24.04
# Configure the repositories.intel.com package repository
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

# Ubuntu22.04
# echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
#   sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Update the package repository meta-data
sudo apt update

# Install the compute-related packages
apt-get install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo libze-dev intel-ocloc

# ## Verify Installation

# #To verify that the kernel and compute drivers are installed and functional, run clinfo:
# clinfo | grep "Device Name"

# #You should see the Intel graphics product device names listed. If they do not appear, ensure you have permissions to access /dev/dri/renderD*. This typically requires your user to be in the render group:
# sudo gpasswd -a ${USER} render
# newgrp render

