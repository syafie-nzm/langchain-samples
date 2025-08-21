# Medical Transcribe demo
![pipeline_diagram](assets/pipeline_diagram.png)

## Whisper ASR model serving (Openvino Whisper.cpp)
### Install and setup OpenVINO and Whisper.cpp (recommended OpenVINO 2024.6)
```bash
bash install-whisper.sh
```
#### Potential error
If you see `E: Unmet dependencies. Try 'apt --fix-broken install' with no packages (or specify a solution).`. Run `sudo apt --fix-broken install`.

### Run Whisper.cpp model serving
```bash
bash run-whisper.sh
```

## Llama3.2 3B model serving (OVMS)
### Install and setup
1. Gain access to [Llama3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) group of models on Hugging Face.
2. Ensure you add the huggingface access token `HUGGINGFACE_TOKEN=<HF_TOKEN>` in `install-llama-ovms.sh`
3. The installation is validated on Ubuntu 22.04 and Ubuntu 24.04.

```bash
bash install-llama-ovms.sh
```
### Run Llama3.2 model serving
```bash
bash run-llama-ovms.sh
```
## Run Streamlit UI
```bash
bash run-streamlit.sh
```
Can access the demo at `http://localhost:8080`. Feel free to use the audio sample `samples/sample.wav` and upload in the demo, or you can record a simulation of medical checkup qna. 