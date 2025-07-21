# Intel XPU Backend for Triton (Intel Flex)
This repository serves as a central hub for the Flex PG Triton XPU Initiative.

The primary goal of this repository is to compile PyTorch and Triton, and to execute Triton tests on the Intel XPU backend.

> [!TIP]
> This README is referenced from the main repository: [Intel-XPU-Backend-For-Triton](https://github.com/intel/intel-xpu-backend-for-triton)

# Compatibility
- Operating systems:
  - WSL2
  - [Ubuntu 22.04](http://releases.ubuntu.com/22.04/)
- GPU Cards:
  - [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html) (Supported, but not Tested)
  - [Intel® Data Center Flex Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html) (Supported, but not Tested)
  - [Intel® Arc A770](https://www.intel.com/content/www/us/en/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)
- GPU Drivers:
  - Latest [Long Term Support (LTS) Release](https://dgpu-docs.intel.com/driver/installation.html)
  - Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html)
  - [Intel® Arc Control](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)
- Toolchain:
  - Latest [Intel® Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)
  
# Prerequisites Installation
## GPU Drivers Installation
GPU Drivers:
  - Latest [Long Term Support (LTS) Release](https://dgpu-docs.intel.com/driver/installation.html)
  - Latest [Rolling Release](https://dgpu-docs.intel.com/driver/installation-rolling.html)
  - Ubuntu: [Client GPU](https://dgpu-docs.intel.com/driver/client/overview.html#selecting-the-right-operating-system-version)
  - [Optional] Windows: [Intel® Arc Control](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

## Dependencies Installation
> [!NOTE]
> The Intel® oneAPI Base Toolkit (Version 2025.0.0) is incompatible with PyTorch and Triton versions released prior to November 12, 2024.
>
> _As of November 12, 2024, [archive/setup.sh](archive/setup.sh) and [archive/setup_nobuild.sh](setup_nobuild.sh) in archive folder will no longer be supported._

Latest release of prerequisites: [Intel® Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-lin=offline&dl-essentials-os=linux)
> It is recommended to initialize oneAPI variables in the ~/.bashrc file so that they are automatically set whenever the terminal starts.
> ```
> # Remember to change <toolkit-version> to the correct version for example 2025.0.0
> source /opt/intel/oneapi/<toolkit-version>/oneapi-vars.sh >> ~/.bashrc
> source ~/.bashrc
> ```

You may utilize the script [setup.sh](setup.sh) to install the latest prerequisite automatically.
```
./setup.sh
```

# Manual Steps
## Install PyTorch and Triton from nightly wheels
Currently, Intel® XPU Backend for Triton* requires a special version of PyTorch and can be installed from nightly wheels. Navigate to the [nightly wheels](https://github.com/intel/intel-xpu-backend-for-triton/actions/workflows/nightly-wheels.yml), select the most recent successful run on the top of the page and download an artifact for the corresponding Python version
```
pip install *.whl
```

## Install PyTorch and Triton from source
Currently, Intel® XPU Backend for Triton* requires a special version of PyTorch and both need to be compiled at the same time.
Below are the steps to build from source:
1) Clone this repository:
```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton
```
2) To avoid potential conflicts with installed packages it is recommended to create and activate a new Python virtual environment:
```
python -m venv .venv --prompt triton
source .venv/bin/activate
```
3) Compile and install PyTorch:
```
scripts/install-pytorch.sh --source
```
4) Compile and install Intel® XPU Backend for Triton:
```
scripts/compile-triton.sh
```
## Triton Test
To ensure that the virtual environment with PyTorch and Triton installed is functioning correctly, you can test it by running one of the tutorials from the intel-xpu-backend-for-triton repository using the following command:
```
cd python/tutorials
python 01-vector-add.py
```
To perform a comprehensive Triton test that includes various types of tests such as unit tests, core tests, tutorials, and more, use the following command:
```
scripts/test-triton.sh
```
> Please note that there are some flags needed to initiate the test case, you may refer to [intel-xpu-backend-for-triton/scripts/test-triton.sh](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/scripts/test-triton.sh).

# Automated Steps
> [!IMPORTANT]
> Users can choose to utilize the developed workflows in this GitHub Actions to automate tasks.
> There are two workflows available: [daily_build.yml](.github/workflows/daily_build.yml) and [daily_runtest.yml](.github/workflows/daily_runtest.yml).

## Usage
To utilize these workflows on your machine, you will need to set up a [Self-Hosted Runner](https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners#adding-a-self-hosted-runner-to-a-repository) from GitHub and replace the existing runner name with the name of your newly created runner in both [daily_build.yml](.github/workflows/daily_build.yml) and [daily_runtest.yml](.github/workflows/daily_runtest.yml).
```
#Existing runner
runs-on: a770-lab
#Your runner
runs-on: <your runner>
```
 
## Daily_Build Workflow
This workflow is used to build and upload Pytorch and Triton wheels as artifact.

[daily_build.yml](.github/workflows/daily_build.yml) covers several task:
1) Install Pytorch and Triton from source.
2) Build their respective wheels.
3) Upload the wheels as GitHub Artifacts into [Daily_Build](https://github.com/intel-sandbox/personal.flexpg.tritonxpu-common/actions/workflows/daily_build.yml).

## Daily_Runtest WorkFlow
This workflow is developed to run the test automatically with different build.

[daily_runtest.yml](.github/workflows/daily_runtest.yml) covers several task:
1) Clones the intel-xpu-backend-for-triton repository.
2) Creates a virtual environment named .prebuild.
3) Downloads and installs the latest prebuild from the nightly wheels into .prebuild.
4) Runs Triton tests in the .prebuild environment.
5) Creates a virtual environment named .devbuild.
6) Downloads and installs the latest devbuild from the artifacts uploaded at [daily_build.yml](.github/workflows/daily_build.yml).
7) Runs Triton tests in the .devbuild environment.
8) Records and pushes the report, log file, and result CSV file to this repository.

Additionally, the repeat_test.sh script in the workflow requires four arguments, as shown below:
```
./repeat_test.sh <name of venv> <working directory> <number of test to repeat> "<flags needed by test-triton.sh>"
```
Users will need to change these arguments accordingly in the yaml file.
