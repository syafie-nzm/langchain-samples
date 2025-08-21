#!/bin/bash

HUGGINGFACE_TOKEN=
DEVICE="GPU"
LLAMA_MODEL="meta-llama/Llama-3.2-3B-Instruct"
UBUNTU_VERSION=$(lsb_release -rs)

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Please set the HUGGINGFACE_TOKEN variable"
    exit 1
fi

# One-time installs
if [ "$1" == "--skip" ]; then
	echo "Skipping dependencies"

else
    # neo/opencl drivers 24.45.31740.9
    mkdir neo
    cd neo
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb
    sudo dpkg -i *.deb
    # sudo apt install ocl-icd-libopencl1
    cd ..

fi

python3 -m venv ovms_venv
source ovms_venv/bin/activate
python -m pip install --upgrade pip

# Install dependencies
sudo apt update -y && sudo apt install -y libxml2 curl
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/requirements.txt -o ovms_requirements.txt
pip install "Jinja2==3.1.6" "MarkupSafe==3.0.2" "huggingface_hub" "streamlit" "librosa"
pip install -r ovms_requirements.txt

# Download OVMS
export PATH=$PATH:${PWD}/ovms/bin
if command -v ovms &> /dev/null; then
    echo "OpenVINO Model Server (OVMS) is already installed."
else
    echo "Downloading OpenVINO Model Server (OVMS)..."
    if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
        # Ubuntu 24.04
        echo "Downloading for Ubuntu 24.04..."
        wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu24_python_on.tar.gz
        tar -xzvf ovms_ubuntu24_python_on.tar.gz

    elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
        echo "Downloading for Ubuntu 22.04..."    
        # Ubuntu 22.04
        wget https://github.com/openvinotoolkit/model_server/releases/download/v2025.1/ovms_ubuntu22_python_on.tar.gz
        tar -xzvf ovms_ubuntu22_python_on.tar.gz
    else
        echo "Error: Unsupported Ubuntu version: $UBUNTU_VERSION"
        echo "Only Ubuntu 22.04 and 24.04 are supported."
        exit 1
    fi
fi

huggingface-cli login --token $HUGGINGFACE_TOKEN

curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/1/demos/common/export_models/export_model.py -o export_model.py
mkdir -p models

echo "Creating OpenVINO optimized model files for LLAMA on device: $DEVICE"
if [ "$DEVICE" == "CPU" ]; then
    python export_model.py text_generation --source_model $LLAMA_MODEL --config_file_path models/config.json --model_repository_path models --target_device $DEVICE --weight-format fp16 --kv_cache_precision u8 --pipeline_type LM --overwrite_models

elif [ "$DEVICE" == "GPU" ]; then
    python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $DEVICE --cache 2 --pipeline_type LM --overwrite_models

elif [ "$DEVICE" == "NPU" ]; then
    python export_model.py text_generation --source_model $LLAMA_MODEL --weight-format int4 --config_file_path models/config.json --model_repository_path models --target_device $DEVICE --max_prompt_len 1500 --pipeline_type LM --overwrite_models
else
    echo "Invalid DEVICE value. Please set it to GPU, CPU or NPU in .env file."
    exit 1
fi