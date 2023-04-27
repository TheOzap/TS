help:
	@cat Makefile

# to see the logs of a container just do $>docker logs -t wtf_docker_name
TIME=$(shell date +'%Y-%m-%d')
HERE?=$(shell dirname `pwd`)

GPU?=all
DOCKER_FILE=Dockerfile


UID=$(shell id -u)
GID=$(shell id -g)
USER=$(shell whoami)
DISPLAY_DOCKER=$DISPLAY
TAG=yfinance
VERSION=1.0

build:
	docker build -t ${TAG}:${VERSION} --build-arg NB_USER=$(USER) --build-arg NB_UID=$(UID) -f $(DOCKER_FILE) .

bash: build
	docker run --init --privileged -e USER=$(USER) -e GID=$(GID) \
		-e UID=$(UID) \
		-p 8888:8888 \
		-e JUPYTER_ENABLE_LAB=yes \
		-e JUPYTER_TOKEN=docker \
		-p 0.0.0.0:6006:6006 \
		-e CUDA_DEVICE_ORDER=PCI_BUS_ID \
		-e CUDA_VISIBLE_DEVICES=0 \
		-e TF_FORCE_GPU_ALLOW_GROWTH=true \
		-it -v "$(HERE):/workspace" \
		-h "" -e DISPLAY=$(DISPLAY) \
		-w /workspace ${TAG}:${VERSION} bash \
	

# docker run --init --privileged --gpus=all -e USER=$(USER) -e GID=$(GID) -e UID=$(UID) -p 0.0.0.0:6006:6006 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES=0 -e TF_FORCE_GPU_ALLOW_GROWTH=true -it -v "$(HERE):/workspace" -h "" -e DISPLAY=$(DISPLAY) -v "/tmp/.X11-unix:/tmp/.X11-unix" -w /workspace ${TAG}:${VERSION} bash

# tensorboard --logdir=Alfred/Results/experiments/ --host 0.0.0.0 &
# jupyter notebook --ip 0.0.0.0 --allow-root
