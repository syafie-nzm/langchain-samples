#!/bin/bash

dpkg -s sudo &> /dev/null
if [ $? != 0 ]
then
	DEBIAN_FRONTEND=noninteractive apt update
	DEBIAN_FRONTEND=noninteractive apt install sudo -y
fi

source activate-conda.sh

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping qna dependencies"
	activate_conda
else
	echo "Installing qna dependencies"
	sudo DEBIAN_FRONTEND=noninteractive apt update
	sudo DEBIAN_FRONTEND=noninteractive apt install -y curl git ffmpeg vim portaudio19-dev build-essential wget -y

	CUR_DIR=`pwd`
        cd /tmp
	miniforge_script=Miniforge3-$(uname)-$(uname -m).sh
	[ -e $miniforge_script ] && rm $miniforge_script
	wget "https://github.com/conda-forge/miniforge/releases/latest/download/$miniforge_script"
	bash $miniforge_script -b -u
	# used to activate conda install
	activate_conda
	conda init
	cd $CUR_DIR

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

	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html --create-dirs -o ./docs/ovms_what_is_openvino_model_server.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_metrics.html -o ./docs/ovms_docs_metrics.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_streaming_endpoints.html -o ./docs/ovms_docs_streaming_endpoints.html
	curl https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_target_devices.html -o ./docs/ovms_docs_target_devices.html
fi


echo "Installing qna"
# Default Python to Ubuntu 22.04.5
conda create -n langchain_qna_env python=3.11.11 -y # for a specific version
conda activate langchain_qna_env
echo 'y' | conda install pip

pip install -r requirements.txt
