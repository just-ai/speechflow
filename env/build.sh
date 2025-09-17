#!/bin/bash

if 0
then
  IMG_NAME="torch_$(date '+%d%m%Y').sif"

  echo "### Build singularity container ###"
  sudo env "PATH=$PATH" singularity build --sandbox $IMG_NAME env/Singularityfile

  echo "### Create overlay file ###"
  singularity overlay create --size 24576 $IMG_NAME
else
  IMG_NAME="torch_$(date '+%d%m%Y').img"

  echo "### Build singularity container ###"
  sudo env "PATH=$PATH" singularity build --sandbox $IMG_NAME env/Singularityfile

  sudo touch $IMG_NAME/usr/bin/nvidia-smi
  sudo touch $IMG_NAME/usr/bin/nvidia-debugdump
  sudo touch $IMG_NAME/usr/bin/nvidia-persistenced
  sudo touch $IMG_NAME/usr/bin/nvidia-cuda-mps-control
  sudo touch $IMG_NAME/usr/bin/nvidia-cuda-mps-server
fi

echo "### Install python packages ###"
singularity run -w --no-home -B .:/src --pwd /src $IMG_NAME pip install -r requirements.txt
singularity run -w --no-home -B .:/src --pwd /src $IMG_NAME pip install fairseq==0.12.2 --no-deps
singularity run -w --no-home -B .:/src --pwd /src $IMG_NAME sh libs/install.sh
