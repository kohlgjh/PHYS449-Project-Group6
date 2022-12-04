# PHYS449-Project-Group6

Authors: Delaney Dyment, Kohl Hedley

Reference paper: https://doi.org/10.1140/epjs/s11734-021-00203-z

Link to final presentation: https://docs.google.com/presentation/d/1i9LuOoyhKChw1HjpDN-0-p6UwPUa6W-4Q0HZWyI14TI/edit#slide=id.g1ae45da4bd8_2_8

Goal: To recreate the results of the reference paper to gain a better understanding of machine learning in physics. The reference paper explored machine learning as a way to automate classification of exoplanets into 3 categories [non-habitable, mesoplanets, psychroplanets]

Dependencies (see requirements.txt): -Pytorch -Numpy -matplotlib -Pandas

To run: python main.py --case #

Where # is an integer between 1 and 8 (inclusive).
Additional, optional arguments:

--verbose: True/False for high/low verbosity during training. Defaults to True

--seed: integer to seed random number generation. Defaults to 1234

--params: relative path to JSON parameter file. Defaults to input/params.json

--device: cuda or cpu. Device that pytorch should use during training.

## Overview:
The focus is on comparing RWNN and Vanilla networks across each of the 8 feature cases. The code base is setup to handle an integer between 1 and 8 inclusive that specifies what feature case to use. Each case has hyper parameters associated with it, including shape of neural network, and whether or not to normalize data, along with standard hyperparameters. 

Structure of Repository:
---
### Folders
data/processed: folder containing the processed data (text-to-number conversion, and in-fill of empty entries - from author's repo) for each of the eight feature cases

input: folder containing the JSON hyperparameter file. Divided into case number so each feature case has its own hyperparameters.

notebooks: folder containing jupyter notebooks that were used solely for testing purposes

results: folder containing a subfolder for each case. Stores plots of loss and accuracy for each feature case during most recent training.

src: folder containing all source code (other than main.py). Subfolder model contains scripts for RWNN, vanilla networks and the base neural network (model.py) that both models use. plotting.py handles the generation of plots. process_data.py handles the processing of data (upsampling/downsampling, splitting into train/test, and normalization).

### Scripts
model.py: Contains the neural net class structure that both the RWNN and Vanilla classes use

RWNN.py: contains the RWNN class. Utilizes model from model.py. Handles generating subsets from data, calculating accuracies, training, and testing. Utlizes subsets of data to train. Trains on one subset for num_epochs, then moves on to the next subset, so on and so forth until it has gone through passed number of iterations.

vanilla.py: contains the Vanilla class. Utilizes model from model.py. Handles calculating accuracies, training, and testing. This model does not use subsets of data when training.

process_data.py: reads in the csv file corresponding to the case_num passed. Splits data by habitability label. Then equally samples from each subset so that all three habitability labels are equally represented. Concatenates the samplings, shuffles them so they are not grouped, and splits into train/test data. Using the total combination of training and testing data, the min and max of each column is calcualted so that the data can be normalized to within [0,1] for each column. This is optional. Finally, labels are separated into their own arrays and converted to one-hot vectors.

plotting.py: passed the results of training from both networks and generates an accuracy plot for each RWNN, and Vanilla, and a singular loss plot for both.
