#!/usr/bin/bash

# Job name:
#SBATCH --job-name=$SBATCH_JOB_NAME
#
# Standard output and error log:
#SBATCH --output=$SBATCH_OUTPUT
# Partition:
#SBATCH --partition=SBATCH_PARTITION
# Node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#
# Account:
#SBATCH --account=$SBATCH_ACCOUNT
#
# Wall clock limit:
#SBATCH --time=01:00:00
#
# QoS
#SBATCH --qos=$SBATCH_QOS
#
# GRES for GPUs
#SBATCH --gres=$SBATCH_GRES


echo OUTPUT:    $SBATCH_OUTPUT
echo PARTITION: $SBATCH_PARTITION
echo QOS:       $SBATCH_QOS
echo GRES:      $SBATCH_GRES
echo ACCOUNT:   $SBATCH_ACCOUNT
echo USERNAME:  $USER

source ~/.bashrc 
conda activate multi-node-gpu
srun hostname # each allocated node prints the hostname

allocated_nodes=$(scontrol show hostname $SLURM_JOB_NODELIST)
nodes=${allocated_nodes//$'\n'/ } # replace newlines with spaces
nodes_array=($nodes)
head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Head Node: $head_node
echo Node List: $nodes
export LOGLEVEL=INFO

cwd=$(pwd)
echo "srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node:29500 $cwd/multinode.py 50 10"

srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node:29500 $cwd/multi_node_gpu.py 50 10

