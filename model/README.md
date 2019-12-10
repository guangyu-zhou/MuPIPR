# Model
This folder contains our main MuPIPR model and its variants: MuPIPR-noAux and MuPIPR-CNN, together with example scripts. 

## How to run

First make sure you follow the readme file of the package and satisfy all the prerequisites.

We provide 4 example shell scripts to run different model variants on the two tasks. 

To run, simply go to the script folder of the current directory and run the corresponding script. 

	cd script
	./run_binding_main.sh


## Parameters
To run ```MuPIPR.py```, we need to enable CUDA by setting ```CUDA_VISIBLE_DEVICES=0```. In addition, we also need to provide the following parameters in order:

	- Protein doublets and score file 
	- Protein id to sequence file
	- The starting index of score (from 0) 
	- The result file name
	- Hidden dimension of RCNN 
	- Number of epochs
	- Dimension of pre-trained biLM
	- Max lines of data to read (-1 for not limit)
	

Take ```run_binding_main.sh``` for example:

	CUDA_VISIBLE_DEVICES=0 python MuPIPR.py ../../data/binding_affinity/bpx.3g.txt ../../data/binding_affinity/bpx.seq.txt 4 results/bfx_3G_2l64_50_50_64.txt 50 50 64 -1


## Output and evaluation metrics
By default, all scripts will conduct 5-fold cross-validation. Under each fold, the Mean Squared Error (MSE) and the Pearson Correlation Coefficient (Corr) will be printed upon finishing. 

After 5 fold running, the final average of MSE and Corr will be printed and saved to the result folder. The prediction of each protein doublets will be saved into the record folder.