FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-dev git libgl1 libglib2.0-0 && apt-get clean
WORKDIR /workspace
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.10 -m pip install --no-deps .
ENTRYPOINT ["torchrun"]
