# Arabic-English VQA

A multi-lingual Visual Question Answering (VQA) system that supports both Arabic and English. This project leverages a Pix2Struct model with PyTorch Lightning for training and evaluation, and provides API endpoints for inference through FastAPI. The codebase is designed to be modular and easily customizable to fit your dataset or specific requirements.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Docker Setup](#docker-setup)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## Overview

The **Arabic-English VQA** project is built to address visual question answering across two languages—Arabic and English. It uses state-of-the-art deep learning techniques to extract features from images and process natural language questions, producing accurate answers along with confidence metrics. The modular design, which separates data processing, model training, and inference, makes it easy for developers to adapt or extend the codebase for their own applications.

---

## Features

- **Multi-Lingual Support:**  
  Processes questions in both Arabic and English.
  
- **Deep Learning Architecture:**  
  Based on the Pix2Struct model for conditional generation using both visual and textual inputs.
  
- **Modular Design:**  
  Clearly separated modules for data preprocessing, training, and inference. Key scripts include:
  - `finetune.py` for model training.
  - `batch_inference.py` and API servers (`server.py` and `server_imi.py`) for inference.
  - `utils.py` for dataset handling and training utilities.
  
- **Confidence Estimation:**  
  Inference endpoints compute confidence scores for the generated answers.
  
- **Docker Support:**  
  Pre-built Docker images and instructions are provided for both CPU and GPU deployments.

---

## Requirements

- **Programming Language:** Python 3.6+
- **Key Libraries & Frameworks:**
  - [PyTorch](https://pytorch.org/)
  - [PyTorch Lightning](https://www.pytorchlightning.ai/)
  - [Transformers](https://huggingface.co/transformers/)
  - [FastAPI](https://fastapi.tiangolo.com/)
  - [Uvicorn](https://www.uvicorn.org/)
  - Additional libraries: `wandb`, `numpy`, `nltk`, `pillow`, `pdf2image`, etc. (see [requirements.txt](requirements.txt) for the full list)
- **Hardware:**  
  GPU recommended for training. CPU is supported for inference or small experiments.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/codetogamble/arabic-english-vqa.git
   cd arabic-english-vqa
   ```

2. **Create a Virtual Environment (optional but recommended):**

   ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

   ```
3. **Install Dependencies:**

   ```bash
    pip install -r requirements.txt
   ```


---

## Configuration

- **Programming Language:** Python 3.6+
- **Environment Variables:**
    Create a .env file in the root directory (if needed) to override default settings. Example:
    ```plaintext
    DATA_DIR=/path/to/data
    MODEL_PATH=/path/to/model
    MAX_PATCHES=3072
    MAX_LENGTH=256
    BATCH_SIZE=1
    NUM_GPUS=1
    NUM_EPOCHS=5
    LR=5e-5
    ```
- **Configuration Files:**
Some scripts (e.g., `finetune.py`) use configuration dictionaries to set hyperparameters, which you can modify directly in the file or via environment variables.
- **Hardware:**  
  GPU recommended for training. CPU is supported for inference or small experiments.

---

## Usage
### Training
To train the model using the provided Pix2Struct implementation, run:

```bash
python finetune.py
```
This script will:

- Load training and testing data from JSON files (located in `DATA_DIR`).
- Prepare the datasets using the custom `ImageCaptioningDataset` class from `utils.py`.
- Set up the training pipeline with PyTorch Lightning.
- Save the trained model and processor along with a `meta.json` configuration file that stores parameters like `max_patches` and `max_length`.

### Example Docker command for training:

```bash
docker run -it --rm \
  -v /path/to/your/data:/usr/src/data \
  -e MAX_PATCHES=3072 \
  -e MAX_LENGTH=256 \
  -e NUM_EPOCHS=5 \
  --gpus all --ipc=host \
  neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna:v1 \
  python dist/finetune.py
  ```

### Inference
There are multiple scripts and endpoints available for inference:

- Batch Inference Script:
Use `batch_inference.py` to send a batch of image and question pairs to the API endpoint.

- API Server (Image Inference):
Run the FastAPI server using:

```bash
uvicorn server:app --host 0.0.0.0 --port 8003
```
This starts an endpoint at `/qna` where you can send an image file and one or more questions. The server processes the image with the Pix2Struct model and returns answers with confidence scores.

- API Server (PDF Inference):
Alternatively, use `server_imi.py` to handle PDF inputs at the `/json/qna/pdf` endpoint. This server accepts a base64-encoded PDF along with questions, converts each page to an image, and processes them.

Example using `make_sample_request.py`:

```bash
python make_sample_request.py
```
This script sends a sample PDF and question to the server and prints out the JSON response.

## Docker Setup
### Build Docker Image
Build the Docker image with:

```bash
docker build -t "ns-hrsd-image" .
```
### Setup Docker on Ubuntu
Follow these commands to set up Docker and, optionally, GPU support:

```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo usermod -aG docker ${USER}
su - ${USER}
sudo systemctl enable docker
```

Setup Docker GPU
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
Run Inference with Docker
Run the inference container as follows:

```bash
sudo docker run -p 8003:8003 --gpus all \
  -v /path/to/your/trained/model:/usr/src/model \
  -e MODEL_PATH=/usr/src/model \
  neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna:v1 \
  uvicorn server:app --host 0.0.0.0 --port 8003
```
*Tip:* To use the default model, simply omit the MODEL_PATH environment variable.

## Customization
This project is structured for easy customization:

- Data Pipeline:
Modify the ImageCaptioningDataset class in utils.py to change image pre-processing or text tokenization. Update the data paths and pre-processing logic if you wish to support additional data formats.

- Model Architecture:
The model and processor are loaded from the Hugging Face Transformers library. To experiment with different architectures or fine-tune hyperparameters, adjust the configuration in finetune.py.

- Inference Logic:
The API servers (server.py and server_imi.py) include logic for generating answers and computing confidence scores. You can update these sections to customize how responses are generated or formatted.

- Language Support:
To extend the system to support additional languages, update the text pre-processing steps and adjust the tokenization in the processor initialization (see utils.py and the server files).

## CUDA and NVIDIA Drivers Customization

To change the CUDA version or specify a different NVIDIA driver version in the Dockerfile, follow these steps:

1. **Modify the Base Image:**
   Update the Dockerfile’s base image to one that includes your desired CUDA version. For example, change:
   ```dockerfile
   FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04
   ```
to your preferred version (e.g., 11.2, 11.7, etc.).

2. **Install Specific NVIDIA Drivers:** 
If you need a particular NVIDIA driver version, ensure that the host has the required drivers. For example, add a command such as:

```bash
sudo apt-get update && apt-get install -y --no-install-recommends nvidia-driver-460
```
Replace nvidia-driver-460 with the desired driver package.

Verify Compatibility: Confirm that the chosen CUDA version is compatible with your specified NVIDIA driver version. Refer to the [NVIDIA CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for details.

Rebuild the Docker Image: After updating the Dockerfile, rebuild your Docker image:

```bash
docker build -t your-image-name .
```

## Creating license file
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Copy the output of the above command and set it as value in lic.py
This key is used to encrypt and decrypt the license file and thus is critical that it should be different for different customers and saved separately.

export this key in an environment variable called ENCRYPTION_KEY. Edit the python utility script `prepare_license_file.py` to set expiry, activation_date and usage.
```bash
export ENCRYPTION_KEY=<KEY>
python utility_scripts/prepare_license_file.py
```

move the created license file into a folder and mount the folder while running the server like this:
```bash
docker run -it -p 9000:9000 -v /home/shubham/workspace/arabic-english-vqa/license/:/usr/src/app/license ns-arabic-vqa:latest bash -c "cd dist && uvicorn server_lic:app --host 0.0.0.0 --port 9000"
```

## Decoding license file
Once a license file is received from the customer. In order to decode and check the remaining usage and status of the file you can run the following command.
```bash
export ENCRYPTION_KEY=<KEY>
python utility_scripts/decode_lic.py --license license/license.txt
```