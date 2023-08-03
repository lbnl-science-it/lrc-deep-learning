#!/usr/bin/bash
export SBATCH_JOB_NAME=gpu_GTRX
export SBATCH_PARTITION=es1
export SBATCH_QOS=es_normal
export SBATCH_OUTPUT=./log/${SBATCH_JOB_NAME}_%j.log
export SBATCH_GRES=gpu:GRTX2080TI:4

sbatch lrc_slurm_run_GTRX.sbatch
