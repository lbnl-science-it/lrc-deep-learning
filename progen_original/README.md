# progen_original
## About this article
Madani, A., Krause, B., Greene, E.R. et al. Large language models generate functional protein sequences across diverse families. Nat Biotechnol (2023). https://doi.org/10.1038/s41587-022-01618-2
## Source Code
https://doi.org/10.5281/zenodo.7296780
## Checkpoints
https://zenodo.org/record/7309036

## Run:
* Download the pretrained model parameters to [checkpoints](./checkpoints) folder
  * ```wget https://zenodo.org/record/7309036/files/pretrain_progen_full.pth?download=1 -O ./checkpoints/pretrain_progen_full.pth```
* Run this notebook [pytorch_training_new.ipynb](progen_code/pytorch_training_new.ipynb) with training data `mibig_train_new.p` in [`miBIG`](./miBIG/)

