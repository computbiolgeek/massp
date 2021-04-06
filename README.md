# MASSP
MASSP stands for Membrane Association and Secondary Structure Prediction. It is a general method for predicting secondary structure, transmembrane segment and topology, and residue orientation of proteins. MASSP was compared with several classic and state-of-the-art predictors and was shown to perform equally well as the best methods in the field while being generalized to different classes of proteins. If you find MASSP useful in your work, please consider citing the MASSP paper: 
* Li, Bian and Mendenhall, Jeffrey and Capra, John A. and Meiler, Jens, A multi-task deep-learning system for predicting membrane associations and secondary structures of proteins, 2020, doi:10.1101/2020.12.02.409045

## Requirements
MASSP depends the following external software packages to generate multiple sequence alignments (MSA) and position-specific-scoring matrices.
  * HHblits
  * The BioChemical Library (BCL)
It was developed and testing under the following Python/Keras/TensforFlow environment.
  * Keras 2.3.1 with TensorFlow 2.0.0 as the backend.
  * Python 3.7

## Obtaining MASSP
Clone MASSP to a local directory.
```bash
git clone https://github.com/computbiolgeek/massp.git
```

## Running MASSP
### Install and set up HHBlits
To be added ...

### Install and set up BCL
To be added ...

### Install TensorFlow and Keras
There are many resources out there that one can follow to install TensorFlow and Keras. We found it easiest to install them with the Anaconda Python distribution.
1. Get the Python 3.7 version [Anaconda 2019.10](https://www.anaconda.com/distribution/) for Linux installer. 
2. Follow the instructions [here](https://docs.anaconda.com/anaconda/install/linux/) to install it.
3. Open anaconda-navigator from the comand line. Go to Environments and search for keras and tensorflow, install all the matched libraries.

Alternatively, one can create a conda environment to use Keras and TensorFlow, i.e.:
```bash
# create conda environment for deep learning/neural networks
conda create --name massp python=3.7
conda activate massp

# install GPU version if you have a GPU already configured for deep learning
conda install keras-gpu==2.3.1
conda install tensorflow-gpu==2.0.0

# otherwise install the CPU version
conda install keras==2.3.1
conda install tensorflow==2.0.0
```

## References
To be added ...
