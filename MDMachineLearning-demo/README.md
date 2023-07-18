forked from [Imay-King/MDMachineLearning](https://github.com/Imay-King/MDMachineLearning)

# MDMachineLearning

## The example codes for modified autoencoder in paper.
### Introduction
We've put the Alanine 13 data(csv files) in Ala,all the data processing and training codes are in the jupyter Notebook. If you want to try other protein data, please input files to the model via correct format. 


Note: We use MDAnalysis(https://www.mdanalysis.org/) to pre-process the dcd/pdb file.

### Requirements
Python 3.6

Keras 2.2.4

Tensorflow 1.12.0

### Reference
N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations. J. Comput. Chem. 32 (2011), 2319-2327, doi:10.1002/jcc.21787. PMCID:PMC3144279 


### Run on [Lawrencium Cluster](https://it.lbl.gov/service/scienceit/high-performance-computing/)
* #### Create conda environment
```
conda create -n MDMachineLearning python=3.6
conda activate MDMachineLearning
```

* #### Add conda environment as a Jupyter kernel
```
conda install ipykernel
python -m ipykernel install --user --name MDMachineLearning --display-name "MDMachineLearning"
```

* #### Install packages
```
pip3 install -r requirements.txt
```

* #### Run notebook [ModifiedAE-demo.ipynb](./ModifiedAE-demo.ipynb)