# MASSP
MASSP stands for Membrane Association and Secondary Structure Prediction. It is a general method for predicting secondary structure, transmembrane segment and topology, and residue orientation of proteins. MASSP was compared with several classic and state-of-the-art predictors and was shown to perform equally well as the best methods in the field while being generalized to different classes of proteins. If you find MASSP useful in your work, please consider citing the MASSP paper: 
* Li, Bian and Mendenhall, Jeffrey and Capra, John A. and Meiler, Jens, A multi-task deep-learning system for predicting membrane associations and secondary structures of proteins, 2020, doi:10.1101/2020.12.02.409045

## Requirements
MASSP depends the following external software packages to generate multiple sequence alignments (MSA) and position-specific-scoring matrices.
  * HHblits
  * The BioChemical Library (BCL)
It was developed and tested under the following Python/Keras/TensforFlow environment.
  * Keras 2.3.1 with TensorFlow 2.0.0 as the backend.
  * Python 3.7

## Obtaining MASSP
Clone MASSP to a local directory.
```bash
git clone https://github.com/computbiolgeek/massp.git
```

## Running MASSP
### Install and configure HHblits
Follow the instructions [here](https://github.com/soedinglab/hh-suite) to install and configure HHblits. We used the UniRef20 database in this work, which can be downloaded [here](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/). Then export the following environment variables for MASSP to run.
```bash
# path to sequence database
export SEQ_DB="path to local UniRef20 database"

# path to hhblits binary
export HHBLITS="path to hhblits binary"

# path to reformat.pl
export REFORMAT="path to reformat.pl"
```

### Install and configure the BCL
Follow the instructions [here](http://www.meilerlab.org/index.php/bclcommons/show/b_apps_id/1) to obtain and install the BCL locally. Then export the BCL for MASSP to run.
```bash
exprot BCL="path to the bcl executable"
```

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

### Run examples
Change to the `examples/` subdirectory to run the following two examples. You do not need to have HHblits or the BCL configured to run these examples. The position-specific-scoring matrices for them have been pre-computed and provided. However, you do need to configure the Python/Keras/TensorFlow environment. In addition, lines 27-30 in `massp.py` will needed to be commented out.
```bash
# 2yevB
python ../massp/massp.py -q 2yevB.fasta -p 2yevB.pssm --cnn-model ../models/final_cnn_model.h5 --lstm-model ../models/final_lstm_model.h5 --tokenizer ../models/tokenizer.pickle --output-prefix 2yevB

# 4c00A
python ../massp/massp.py -q 4c00A.fasta -p 4c00A.pssm --cnn-model ../models/final_cnn_model.h5 --lstm-model ../models/final_lstm_model.h5 --tokenizer ../models/tokenizer.pickle --output-prefix 4c00A
```
Then run the following command to compute performance metrics.
```bash
# remove unresolved residues in the structure
python ../scripts/process_sse_prediction.py ../ciphipsi/2yevB.ciphipsi 2yevB_predicted_sse.fasta 2yevB
python ../scripts/process_sse_prediction.py ../ciphipsi/4c00A.ciphipsi 4c00A_predicted_sse.fasta 4c00A

# run SOV_refine.pl to compute performance metrics
perl ../scripts/SOV_refine.pl 2yevB_true_sse.fasta 2yevB_processed_sse.fasta
perl ../scripts/SOV_refine.pl 4c00A_true_sse.fasta 4c00A_processed_sse.fasta
```


## References
### MASSP
* Li, Bian and Mendenhall, Jeffrey and Capra, John A. and Meiler, Jens, A multi-task deep-learning system for predicting membrane associations and secondary structures of proteins, 2020, doi:10.1101/2020.12.02.409045
### SOV_refine
* Liu, T. and Z. Wang, SOV_refine: A further refined definition of segment overlap score and its significance for protein structure similarity. Source Code Biol Med, 2018. 13: p. 1.
