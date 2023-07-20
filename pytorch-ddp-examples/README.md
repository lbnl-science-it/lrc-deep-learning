# Run pytorch Distributed Data Parallel(DDP) on Lawrencium Cluster
forked from [pytorch/examples](https://github.com/pytorch/examples)

Code for the DDP tutorial series at https://pytorch.org/tutorials/beginner/ddp_series_intro.html

## Files
* [multi_node_gpu.py](multi_node_gpu.py): DDP on multiple nodes and multiple GPUs using Torchrun
* [lrc_slurm_run.sbatch](lrc_slurm_run.sbatch): Lawrencium cluster slurm script to launch a training job on 2 nodes, each node has 4 of GPUs(NVidia A40)
* [run_gpuA40_node.sh](run_gpuA40_node.sh): Set up slurm environment variables, and submit slurm job

## Run DDP on multiple nodes & GPUs on [Lawrencium Cluster](https://it.lbl.gov/service/scienceit/high-performance-computing/)
### Request an [Interactive Jupyter Server](https://it.lbl.gov/resource/hpc/for-users/hpc-documentation/open-ondemand/jupyter-server/) on ES1 GPU partition from [Lawrencium Open OnDemand](https://lrc-ondemand.lbl.gov)
* Connect to Jupyter and open a terminal

### Create conda environment
```
conda create -n multi-node-gpu python=3.8
conda activate multi-node-gpu
```

### Add conda environment as a Jupyter kernel (optional)
```
conda install ipykernel
python -m ipykernel install --user --name multi-node-gpu --display-name "multi-node-gpu"
```

### Install packages
```
pip3 install -r requirements.txt
```

### Run the training job on multiple nodes and multiple GPUs
* request 2 nodes on ES1 partition, each node has 4 GPUs
* use `torchrun` to train a model on __8 GPUs__ 

Replace `ac_XYZ` with __your Lawrencium slurm account__.
```
# ac_XYZ needs to be replaced with your Lawrencium slurm account
export SBATCH_ACCOUNT=ac_XYZ
# submit the slurm job
sh ./run_gpuA40_node.sh
```

Outputs in [./log](log):
* training data size: 2048
* epochs: 50
* save the model every 10 epochs as `snapshot.pt` 
* node1: [GPU0, GPU1, GPU2, GPU3]
* node2: [GPU4, GPU5, GPU6, GPU7]

```
OUTPUT: ./log/gpu_A40_%j.log
PARTITION: es1
QOS: es_normal
GRES: gpu:A40:4
ACCOUNT: ac_XYX
USERNAME: user_XYZ
n0000.es1
n0001.es1
Head Node: n0000.es1
Node List: n0000.es1 n0001.es1
srun torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id 7926 --rdzv_backend c10d --rdzv_endpoint n0000.es1:29500 /global/scratch/users/user_XYZ/GitHub/pytorch-examples/distributed/ddp-tutorial-series/lrc/multinode.py 50 10
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
master_addr is only used for static rdzv_backend and when rdzv_endpoint is not specified.
INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
  entrypoint       : /global/scratch/users/user_XYZ/GitHub/pytorch-examples/distributed/ddp-tutorial-series/lrc/multi_node_gpu.py
  min_nodes        : 2
  max_nodes        : 2
  nproc_per_node   : 4
  run_id           : 10889
  rdzv_backend     : c10d
  rdzv_endpoint    : n0000.es1:29500
  rdzv_configs     : {'timeout': 900}
  max_restarts     : 0
  monitor_interval : 5
  log_dir          : None
  metrics_cfg      : {}

INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_5dtsrpgd/10889_zhbb_9hb
INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
INFO:torch.distributed.launcher.api:Starting elastic_operator with launch configs:
  entrypoint       : /global/scratch/users/user_XYZ/GitHub/pytorch-examples/distributed/ddp-tutorial-series/lrc/multi_node_gpu.py
  min_nodes        : 2
  max_nodes        : 2
  nproc_per_node   : 4
  run_id           : 10889
  rdzv_backend     : c10d
  rdzv_endpoint    : n0000.es1:29500
  rdzv_configs     : {'timeout': 900}
  max_restarts     : 0
  monitor_interval : 5
  log_dir          : None
  metrics_cfg      : {}

INFO:torch.distributed.elastic.agent.server.local_elastic_agent:log directory set to: /tmp/torchelastic_xoj2jdhn/10889_c62xidfb
INFO:torch.distributed.elastic.agent.server.api:[default] starting workers for entrypoint: python
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous'ing worker group
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous complete for workers. Result:
  restart_count=0
  master_addr=n0000.es1
  master_port=57850
  group_rank=0
  group_world_size=2
  local_ranks=[0, 1, 2, 3]
  role_ranks=[0, 1, 2, 3]
  global_ranks=[0, 1, 2, 3]
  role_world_sizes=[8, 8, 8, 8]
  global_world_sizes=[8, 8, 8, 8]

INFO:torch.distributed.elastic.agent.server.api:[default] Starting worker group
INFO:torch.distributed.elastic.agent.server.local_elastic_agent:Environment variable 'TORCHELASTIC_ENABLE_FILE_TIMER' not found. Do not start FileTimerServer.
INFO:torch.distributed.elastic.multiprocessing:Setting worker0 reply file to: /tmp/torchelastic_5dtsrpgd/10889_zhbb_9hb/attempt_0/0/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker1 reply file to: /tmp/torchelastic_5dtsrpgd/10889_zhbb_9hb/attempt_0/1/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker2 reply file to: /tmp/torchelastic_5dtsrpgd/10889_zhbb_9hb/attempt_0/2/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker3 reply file to: /tmp/torchelastic_5dtsrpgd/10889_zhbb_9hb/attempt_0/3/error.json
INFO:torch.distributed.elastic.agent.server.api:[default] Rendezvous complete for workers. Result:
  restart_count=0
  master_addr=n0000.es1
  master_port=57850
  group_rank=1
  group_world_size=2
  local_ranks=[0, 1, 2, 3]
  role_ranks=[4, 5, 6, 7]
  global_ranks=[4, 5, 6, 7]
  role_world_sizes=[8, 8, 8, 8]
  global_world_sizes=[8, 8, 8, 8]

INFO:torch.distributed.elastic.agent.server.api:[default] Starting worker group
INFO:torch.distributed.elastic.agent.server.local_elastic_agent:Environment variable 'TORCHELASTIC_ENABLE_FILE_TIMER' not found. Do not start FileTimerServer.
INFO:torch.distributed.elastic.multiprocessing:Setting worker0 reply file to: /tmp/torchelastic_xoj2jdhn/10889_c62xidfb/attempt_0/0/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker1 reply file to: /tmp/torchelastic_xoj2jdhn/10889_c62xidfb/attempt_0/1/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker2 reply file to: /tmp/torchelastic_xoj2jdhn/10889_c62xidfb/attempt_0/2/error.json
INFO:torch.distributed.elastic.multiprocessing:Setting worker3 reply file to: /tmp/torchelastic_xoj2jdhn/10889_c62xidfb/attempt_0/3/error.json
[GPU2] Epoch 0 | Batchsize: 32 | Steps: 8[GPU1] Epoch 0 | Batchsize: 32 | Steps: 8[GPU3] Epoch 0 | Batchsize: 32 | Steps: 8


[GPU0] Epoch 0 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 0 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 0 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 0 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 0 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 1 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 1 | Batchsize: 32 | Steps: 8[GPU1] Epoch 1 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 2 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 3 | Batchsize: 32 | Steps: 8[GPU6] Epoch 3 | Batchsize: 32 | Steps: 8

[GPU3] Epoch 3 | Batchsize: 32 | Steps: 8[GPU0] Epoch 3 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 3 | Batchsize: 32 | Steps: 8[GPU1] Epoch 3 | Batchsize: 32 | Steps: 8

[GPU5] Epoch 3 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 3 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 4 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 5 | Batchsize: 32 | Steps: 8[GPU4] Epoch 5 | Batchsize: 32 | Steps: 8

[GPU1] Epoch 5 | Batchsize: 32 | Steps: 8[GPU0] Epoch 5 | Batchsize: 32 | Steps: 8

[GPU7] Epoch 5 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 5 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 5 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 5 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 6 | Batchsize: 32 | Steps: 8[GPU7] Epoch 6 | Batchsize: 32 | Steps: 8

[GPU3] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 6 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 7 | Batchsize: 32 | Steps: 8[GPU3] Epoch 7 | Batchsize: 32 | Steps: 8

[GPU5] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 7 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 8 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 9 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 10 | Batchsize: 32 | Steps: 8
Epoch 9 | Training snapshot saved at snapshot.pt
[GPU0] Epoch 10 | Batchsize: 32 | Steps: 8
Epoch 9 | Training snapshot saved at snapshot.pt
[GPU4] Epoch 10 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 11 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 12 | Batchsize: 32 | Steps: 8[GPU6] Epoch 12 | Batchsize: 32 | Steps: 8

[GPU3] Epoch 12 | Batchsize: 32 | Steps: 8[GPU2] Epoch 12 | Batchsize: 32 | Steps: 8[GPU1] Epoch 12 | Batchsize: 32 | Steps: 8


[GPU5] Epoch 12 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 12 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 12 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 13 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 14 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 15 | Batchsize: 32 | Steps: 8[GPU4] Epoch 15 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 15 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 16 | Batchsize: 32 | Steps: 8[GPU5] Epoch 16 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 16 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 17 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 18 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 19 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 20 | Batchsize: 32 | Steps: 8
Epoch 19 | Training snapshot saved at snapshot.pt
[GPU4] Epoch 20 | Batchsize: 32 | Steps: 8
Epoch 19 | Training snapshot saved at snapshot.pt
[GPU0] Epoch 20 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 21 | Batchsize: 32 | Steps: 8[GPU5] Epoch 21 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 21 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 22 | Batchsize: 32 | Steps: 8[GPU3] Epoch 22 | Batchsize: 32 | Steps: 8

[GPU5] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 22 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 23 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 24 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 24 | Batchsize: 32 | Steps: 8[GPU3] Epoch 24 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 24 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 24 | Batchsize: 32 | Steps: 8[GPU4] Epoch 24 | Batchsize: 32 | Steps: 8

[GPU7] Epoch 24 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 24 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 25 | Batchsize: 32 | Steps: 8[GPU1] Epoch 25 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 25 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 26 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 27 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 28 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 29 | Batchsize: 32 | Steps: 8[GPU4] Epoch 29 | Batchsize: 32 | Steps: 8

[GPU5] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 29 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 30 | Batchsize: 32 | Steps: 8[GPU6] Epoch 30 | Batchsize: 32 | Steps: 8[GPU7] Epoch 30 | Batchsize: 32 | Steps: 8


[GPU1] Epoch 30 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 30 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 30 | Batchsize: 32 | Steps: 8
Epoch 29 | Training snapshot saved at snapshot.pt
[GPU0] Epoch 30 | Batchsize: 32 | Steps: 8
Epoch 29 | Training snapshot saved at snapshot.pt
[GPU4] Epoch 30 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 31 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 32 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 33 | Batchsize: 32 | Steps: 8[GPU7] Epoch 33 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 33 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 34 | Batchsize: 32 | Steps: 8[GPU5] Epoch 34 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 34 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 35 | Batchsize: 32 | Steps: 8[GPU7] Epoch 35 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 35 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 36 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 37 | Batchsize: 32 | Steps: 8[GPU7] Epoch 37 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 37 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 38 | Batchsize: 32 | Steps: 8[GPU6] Epoch 38 | Batchsize: 32 | Steps: 8

[GPU3] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 38 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 39 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 40 | Batchsize: 32 | Steps: 8[GPU3] Epoch 40 | Batchsize: 32 | Steps: 8[GPU2] Epoch 40 | Batchsize: 32 | Steps: 8


[GPU5] Epoch 40 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 40 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 40 | Batchsize: 32 | Steps: 8
Epoch 39 | Training snapshot saved at snapshot.pt
[GPU0] Epoch 40 | Batchsize: 32 | Steps: 8
Epoch 39 | Training snapshot saved at snapshot.pt
[GPU4] Epoch 40 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 41 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 41 | Batchsize: 32 | Steps: 8[GPU3] Epoch 41 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 41 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 41 | Batchsize: 32 | Steps: 8[GPU5] Epoch 41 | Batchsize: 32 | Steps: 8

[GPU6] Epoch 41 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 41 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 42 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 43 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 44 | Batchsize: 32 | Steps: 8[GPU1] Epoch 44 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 44 | Batchsize: 32 | Steps: 8[GPU6] Epoch 44 | Batchsize: 32 | Steps: 8

[GPU4] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 44 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 45 | Batchsize: 32 | Steps: 8[GPU3] Epoch 45 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 45 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 46 | Batchsize: 32 | Steps: 8[GPU3] Epoch 46 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 46 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU3] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU2] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 47 | Batchsize: 32 | Steps: 8
[GPU4] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU7] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 48 | Batchsize: 32 | Steps: 8[GPU3] Epoch 48 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 48 | Batchsize: 32 | Steps: 8
[GPU0] Epoch 49 | Batchsize: 32 | Steps: 8
[GPU1] Epoch 49 | Batchsize: 32 | Steps: 8[GPU3] Epoch 49 | Batchsize: 32 | Steps: 8

[GPU2] Epoch 49 | Batchsize: 32 | Steps: 8
[GPU6] Epoch 49 | Batchsize: 32 | Steps: 8[GPU7] Epoch 49 | Batchsize: 32 | Steps: 8

[GPU4] Epoch 49 | Batchsize: 32 | Steps: 8
[GPU5] Epoch 49 | Batchsize: 32 | Steps: 8
Epoch 49 | Training snapshot saved at snapshot.pt
Epoch 49 | Training snapshot saved at snapshot.pt
INFO:torch.distributed.elastic.agent.server.api:[default] worker group successfully finished. Waiting 300 seconds for other agents to finish.
INFO:torch.distributed.elastic.agent.server.api:Local worker group finished (SUCCEEDED). Waiting 300 seconds for other agents to finish
INFO:torch.distributed.elastic.agent.server.api:[default] worker group successfully finished. Waiting 300 seconds for other agents to finish.
INFO:torch.distributed.elastic.agent.server.api:Local worker group finished (SUCCEEDED). Waiting 300 seconds for other agents to finish
INFO:torch.distributed.elastic.agent.server.api:Done waiting for other agents. Elapsed: 0.0008671283721923828 seconds
INFO:torch.distributed.elastic.agent.server.api:Done waiting for other agents. Elapsed: 0.003488779067993164 seconds
```