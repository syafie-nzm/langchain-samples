#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WHISPER_DIR="$SCRIPT_DIR/whisper.cpp"
OPENVINO_DIR="/opt/intel/openvino_2024"

source $OPENVINO_DIR/setupvars.sh

source $WHISPER_DIR/models/openvino_env/bin/activate

$WHISPER_DIR/build/bin/whisper-server -m $WHISPER_DIR/models/ggml-base.en.bin -oved GPU --port 5910