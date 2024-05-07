# hrsd-vqa

## Build image
```bash
docker build -t "ns-hrsd-image" .
```

## Run training

```bash
docker run -it --rm \
  -v /path/to/your/data:/usr/src/data \
  -e MAX_PATCHES=1024 \
  -e MAX_LENGTH=512 \
  ns-hrsd-image
  finetune.py
```

## Setup docker GPU

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
