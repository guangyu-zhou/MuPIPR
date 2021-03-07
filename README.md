# MuPIPR - Mutation effect estimation on protein-protein interactions using deep contextualized representation learning
This is the repository for the NAR Genom. Bioinform. paper "Mutation effect estimation on protein-protein interactions using deep contextualized representation learning" (MuPIPR). This repository contains the source code and links to some datasets used in our paper. (to be updated)

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
* **biLM:** contains the pre-trained language models for contextualized amino acid representations.
* **data:** contains the datasets and processing scripts for the two tasks.
* **model:** contains the implementation for MuPIPR.


## Using the pre-trained amino acid language model for contextualized representation
We obtain the corpus to pre-train the BiLSTM language model from the STRING database. A total of 66235 protein sequences of four most frequent species from the SKEMPI database are extracted, i.e. Homo sapiens, Bos taurus, Mus musculus and Escherichia coli. These are the four most frequent species in the SKP1402m dataset.

To serve the pre-trained contextualized embedding model to MuPIPR, please download and unzip the model.zip file in the biLM folder.

## Data processing and model running
Please refer to the readme in the **data** folder and the **model** folder, respectively. 


## Reference
This work has been published in the *NAR Genomics and Bioinformatics* journal.

DOI: https://doi.org/10.1093/nargab/lqaa015  
Bibtex:

    @article{zhou2020mupipr,
        title={Mutation Effect Estimation on Protein–protein Interactions Using Deep Contextualized Representation Learning},
        author={Zhou, Guangyu and Chen, Muhao and Ju, Chelsea and Wang, Zheng and Jiang Jyun-yu and Wang, Wei},
        journal={NAR Genomics and Bioinformatics},
        volume = {2},
        number = {2},
        year = {2020},
        month = {03},
        publisher={Oxford University Press}
    }

## PIPR (ISMB 2019)  
Also check out the follow up work in the Bioinformatics (Procs of ISMB) paper [Multifaceted Protein-Protein Interaction Prediction Based on Siamese Residual RCNN](http://dx.doi.org/10.1093/bioinformatics/btz328), in which we provide an end-to-end neural learning system to predict multifaceted PPI information.  
The released software is available at [muhaochen/seq_ppi](https://github.com/muhaochen/seq_ppi).
