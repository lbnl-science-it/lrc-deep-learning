# progen
language modeling for protein sequences

## Setup
1. Create a GCP pod with tensorflow 1.14.0 image such as `tensorflow/tensorflow:1.14.0-gpu`.
2. Pip install requirements file
3. Patch the `/usr/local/lib/python2.7/dist-packages/tensorflow_estimator/python/estimator/keras.py` (or equivalent, if installed elsewhere) by running 

```patch -b <path_to_tensorflow_estimator_package>/python/estimator/keras.py estimator.patch```

## Training Command
Currently you can train by running `python training.py --tfrecords_dir <PATH> --model_dir <PATH>`. 

## Pretraining Vocabulary
The categories for lines in `vocab.txt`:
- 0 to 1164: keyword ids
- 1165 to 129380: taxonomy ids
- 129381 to 129405: amino acids
- 129406: PAD token

## Fine-tuned lysozyme model vocabulary
Assumptions:
- there are k clusters replacing ctrl codes [0,k-1]
- there is a stop token replacing ctrl code k
- the sample length is 511. all extra tokens are replaced with the original pad token 129406

Ordering of CTRL code to protein family
```
0: PF00959
1: PF01832
2: PF05838
3: PF06737
4: PF16754
5: stop token
129406: pad token
```

## Fine-tuned with jgi bgc data using multiple GPUs on Lawrencium Cluster
*  This fine-tune [example](log/gpu_A40_7node_28gpus.log) ran 2048 epochs on [jgi bgc data](../miBIG/S3_pickle/data-1.pickle) and took approximately 36 hours on 28 GPUs (Nvida A40) on [Lawrencium Cluster](https://it.lbl.gov/service/scienceit/high-performance-computing/)

* Run the fine-tune using pytorch Distributed Data Parallel([DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)) on 32 Nvida A40 GPUs.
  * Replace `ac_XYZ` with __your Lawrencium slurm account__.
```
# ac_XYZ needs to be replaced with your Lawrencium slurm account
export SBATCH_ACCOUNT=ac_XYZ
```
```
# submit the slurm job
sh ./run_gpuA40_node.sh
```
