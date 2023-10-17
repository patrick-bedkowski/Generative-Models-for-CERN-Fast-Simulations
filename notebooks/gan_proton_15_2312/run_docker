#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --exclude=titan2
#SBATCH --gpus=2

gpu_id=$CUDA_VISIBLE_DEVICES
echo $gpu_id
gpu_formated="${gpu_id/,/_}"
docker_name=pbedkows_mpi4py_$gpu_formated

# Check the number of available GPUs
num_gpus=$(nvidia-smi -L | wc -l)
echo "Number of available GPUs: $num_gpus"

# build docker iamge with all the dependencies
docker build -t custom_image_pb -f Dockerfile .

echo "GPU ID"
echo $gpu_id

docker run -it --rm -d --gpus "\"device=2\"" -v /home/pbedkows/:/root/ --name $docker_name -w /root/ --ipc=host custom_image_pb

docker exec $docker_name echo $(pwd)
docker exec $docker_name echo "$(ls)"
#docker exec $docker_name echo $(pip3 list)
docker exec $docker_name python "sdi_gan_proton_neutron_2_outputs.py"

docker stop $docker_name
docker rm $docker_name