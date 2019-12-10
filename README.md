# MuPIPR
This is the repository for "Mutation effect estimation on protein-protein interactions using deep contextualized representation learning" (MuPIPR). This repository contains the source code and links to some datasets used in our paper. (to be updated)

## Pre-requisite

MuPIPR can be run under Linux. The following packages are required: python 3.6, h5py, Tensorflow 1.7 (with GPU support), Keras 2.2.4 and bilm-tf.

## Installing
If you don't have python 3.6, please download from [here.](https://www.python.org/downloads/)

Then you can use ```pip install ``` to install the following packages:

	h5py
	Tensorflow 1.7 (with GPU support)
	Keras 2.2.4

Make sure tensorflow and h5py has been installed successfully before you install bilm-tf. To install bilm-tf, please download the package from [here] (https://github.com/allenai/bilm-tf) and run: `python setup.python`
	

## Contents
* **biLM:** contains the pre-trained contextualized embedding models.
* **data:** contains the datasets and processing scripts for the two tasks.
* **model:** contains the implementation for MuPIPR.


## Using the pre-trained contextualized embeddings model
We obtain the corpus to pre-train the contextualized amino acid encoder from the STRING database. A total of 66235 protein sequences of four most frequent species from the SKEMPI database are extracted, i.e. Homo sapiens, Bos taurus, Mus musculus and Escherichia coli. These are the four most frequent species in the SKP1402m dataset.

To serve the pre-trained contextualized embedding model to MuPIPR, please download and unzip the model.zip file in the biLM folder.

## Data processing and model running
Please refer to the readme in the **data** folder and the **model** folder, respectively. 
