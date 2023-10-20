# DeepThinker 

A generic neural network for data multi-classification

## What is this? 

An implementation of a basic deep learning network to be used for classification of data. This supports multi-class datasets in which a single datapoint can fit one of many classes. The program will automatically attempt to utilize a GPU if present, but defaults back to running on your CPU if none is found. 

## Setup

### Requirements 
- Python 3.10+ (tested on Python 3.11, recommended to use 3.11)
- Libraries (install with pip by running `pip3 install [LIBRARY]`)
	- torch
	- csv
	- torchvision
	- collections
	- numpy
	- pathlib 


### Source Modifications 
- Constants 
	- There are 3 constants that help determine the runtime of the program (found at top of nn.py file)
		- TRAIN
			- enables/disables training mode. This allows you to update the model with new data from a CSV 
			- Set to `True` to run the training loop in addition to the testing loop
		- MODEL_FILE
			- If you want the program to run using an existing compatible model, provide the full path to a .pt file
			- This allows you to run the program with a pre-trained model to cut down on training time or to just simply test the performance of a model. 
		- ITERS
			- Specifies the number of splits you wish to perform on the data set.
			- By splitting the data into smaller chunks, more efficient training/testing is achieved
	- Example setup:
		`TRAIN=True
		 MODEL_FILE="/Users/obiwan/csec620/project1/src/py/model.pt"
		 ITERS = 25` 

- Data Type
	- For ease in reading in non-numeric data, a custom data type must be defined. Provided is an example data type for PCAP data (network traffic captures). 
	- Your data type must implement `__init__`, `__getitem__`, and `__len__` at a minimum, but it is recommended to include additional methods to support quicker parsing and conversion 
- CSV_DATA
	- Be sure to set the CSV_DATA variable to the absolute path to your csv file containing the data you wish to train your model on. 

## 