# hrsd-vqa

## Build image
```bash
docker build -t "ns-hrsd-image" .
```

## setup docker

sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common -y

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo usermod -aG docker ${USER}
su - ${USER}
sudo systemctl enable docker


## Setup docker GPU

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```




## Run training

```bash
docker run -it --rm   -v /home/ubuntu/pix2struct-vqa-dummy-data/edc_dummy_data:/usr/src/data   -e MAX_PATCHES=3072   -e MAX_LENGTH=256 -e NUM_EPOCHS=5   --gpus all --ipc=host  neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna:v1   python dist/finetune.py
```



## Run inference
```bash
docker run -p 80:80 --gpus all -e MODEL_PATH <PATH TO TRAINED MODEL> neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna uvicorn server:app --host 0.0.0.0 --port 80

```
Set the model path to the trained `model` folder. Once training is finished, it can be found inside `training` folder in the data folder. For using the default model, remove the `MODEL_PATH` environment variable from above command.





docker run -it --rm   -v /home/elias/neuralspace/dataset/edc_dummy_data:/usr/src/data   -e MAX_PATCHES=3072 -e NUM_GPUS=2 -e BATCH_SIZE=2  -e MAX_LENGTH=256 -e NUM_EPOCHS=5   --gpus all --ipc=host  neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna:v1   python dist/finetune.py

sudo docker run -p 8003:8003 --gpus all -v /home/elias/neuralspace/dataset/edc_dummy_data/training/model:/usr/src/model -e MODEL_PATH=/usr/src/model neuralspaceacr.azurecr.io/hrsd/ns-hrsd-qna:v1 uvicorn server:app --host 0.0.0.0 --port 8003