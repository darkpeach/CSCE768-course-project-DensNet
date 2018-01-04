# Envrionment:
	* Python 3
	* PyTorch 0.3
	* CUDA 9

# Data
	cullpdb should be found by code

# File Description
	* fasta.py -- generate fasta sequence from cullpdb datasets
	* pssm.sh -- use PSI-BLAST to generate PSSM
	* pssm.py -- interpret PSSM from txt files and build the datasets for training
	* plot.py -- plot loss transition and accuracy transition
	* densenet/
		* run.py -- execute this file could start training and validate
		* data_load.py -- load data using PyTorch DataLoader libray
		* denset_net.py -- densenet model object based on PyTorch