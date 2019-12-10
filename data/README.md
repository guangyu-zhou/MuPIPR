# Dataset 
In this folder, we provide the datasets for two tasks and the scripts for processing. The datasets are provided in the ```binding_affinity``` and ```bsa``` folders, and the processing scripts in Jupyter Notebook with instructions are in the ```scripts``` folder.

## Description

**1. Binding Affinity datasets**. Two datasets generated from the SKEMPI database are used for the *binding affinity task*. The first one is a benchmark dataset extracted by [Xiong et al] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5963940/). We denote it as **SKP1402m**. This dataset contains the changes of binding affinity between wild-type and mutated protein complexes that are experimentally measured. These mutations include single and multiple amino acid substitutions on the protein sequences. This dataset contains 1402 doublets for 114 proteins, among which, 1131 doublets contain single-point mutations, 195 contain double-points mutations, and 76 contain three or more mutations. 
The second dataset is provided by [Geng et al.] (https://www.biorxiv.org/content/10.1101/331280v2), which considers only single-point mutation of dimeric proteins. It contains 1102 doublets for 57 proteins. We denote this dataset as **SKP1102s**. Of these 1102 doublets, the majority (759 doublets) are new entries that are not found in SKP1402m. 
	
Specifically, each dataset contains a score file named \*.ddg.txt and a sequence file named \*.seq.txt. All the experiments need to take such a pair as input. The processed files are provided in the ```binding_affinity``` folder. 

As for the SKP1402m dataset, we seperate into two cases: 
	
- Single mutation only (SKP1402m.single.ddg.txt);
- Multiple mutations (SKP1402m.ddg.txt)

**2. BSA dataset.** We use this dataset for the task of estimating BSA changes. To construct the wild-type pairs and their mutant pairs, we extract protein sequences from PDB, and keep those with only two chains. Sequences with less than 20 amino acids are removed. Here a wild-type pair or a mutant pair refers to the two chains of a protein. We concatenate such two chains of a protein for pairwise sequence comparisons, and retain those with one amino acid substitution. This process produces 2948 doublets. 

To compute the true value of BSA, we first run [DSSP] (https://swift.cmbi.umcn.nl/gv/dssp/DSSP_3.html) to obtain the ASA of the proteins based on the 3D structures provided by PDB. The standard estimation of BSA is calculated by taking the difference between the sum of ASA for the individual chains in a protein complex and the ASA of the protein complex.

Similar as the previous datasets, to conduct experiments on the BSA estimation task, we also need to provide both the score file and the sequence file as input. The two processed files are provided in the ```bsa``` folder. 


## Processing scripts
If you would like to process your own dataset, please refer to the ```script``` folder. The instructions are within the Jupyter notebook and the required inputs are also provided in the ```script``` folder. Specifically, 

* The script for processing SKEMPI datasets is in ```script/process_skempi_v2.ipynb```. 
* The scripts for processing PDB datasets are in ```script/run_dssp.ipynb``` and ```script/process_bsa.ipynb```. 

