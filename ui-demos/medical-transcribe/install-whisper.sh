#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
OPENVINO_DIR="/opt/intel/openvino_2024"

if [ ! -d "$WHISPER_DIR" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
else
    echo "whisper.cpp already exists, skipping clone."
fi

# Setup OpenVINO 2024.6
sudo apt update
sudo apt install -y curl

if [ ! -d "$OPENVINO_DIR" ]; then
    curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64.tgz --output openvino_2024.6.0.tgz
    sudo mkdir -p /opt/intel
    tar -xf openvino_2024.6.0.tgz
    sudo mv l_openvino_toolkit_ubuntu22_2024.6.0.17404.4c0f47d2335_x86_64 /opt/intel/openvino_2024
else
    echo "OpenVINO already installed at $OPENVINO_DIR, skipping download and extraction."
fi

# Install venv and cmake
sudo apt update
sudo apt install -y git python3-venv build-essential cmake

cd $SCRIPT_DIR/whisper.cpp/models
python3 -m venv openvino_env

# Setup venv & download and convert models
bash -c "source openvino_env/bin/activate; \
    python -m pip install --upgrade pip; \
    pip install -r requirements-openvino.txt; \
    ./download-ggml-model.sh base.en; \
    python convert-whisper-to-openvino.py --model base.en"

# Compile whisper.cpp binaries
bash -c "source $OPENVINO_DIR/setupvars.sh; \
    cd ..; \
    pwd; \
    cmake -B build -DWHISPER_OPENVINO=1; \
    cmake --build build -j --config Release"

echo "Remember to setup OpenVINO environment variables by: source $OPENVINO_DIR/setupvars.sh"
