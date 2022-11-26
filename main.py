'''Main entry point into code'''
import argparse, json, sys
sys.path.append('src')
from src.model.RWNN import RWNN
from src.model.vanilla import Vanilla
from plotting import plot_train_test
from process_data import generate_train_and_test
import numpy as np
import torch.nn as nn
import torch

def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="integer 1-8 of what feature set to use")
    parser.add_argument('-v', "--verbose", default="True", help="True/False for high/low verbosity")
    parser.add_argument("--seed", default=1234, help="random seed")
    parser.add_argument("--params", default='input/params.json', help="JSON filepath containing hyperparameters")
    parser.add_argument("--device", default="cpu", help="cuda or cpu as device to run pytorch on")
    return parser.parse_args()

def main(args):
    '''Main code block'''
    verbose = True if args.verbose == "True" else False
    case = int(args.case) # case number stored as int
    
    # hyperparameters from json file
    with open(args.params) as paramfile:
        param = json.load(paramfile)

    # pull parameters for particular feature case
    case_param = param[f'case_{case}']
    input_size = case_param['input_size']
    hidden_size = case_param['hidden_size']
    epochs = case_param['epochs']
    learning_rate = case_param['learning_rate']
    momentum = case_param['momentum']
    iterations = case_param['iterations']

    # generate datasets
    data = generate_train_and_test(case, seed=int(args.seed))

    # creation of models
    rwnn = RWNN(data, epochs, iterations, learning_rate, momentum, verbose, input_size, hidden_size, args.device)
    vanilla = Vanilla(data, epochs, iterations, learning_rate, momentum, verbose, input_size, hidden_size, args.device)
    
    rwnn_obj_vals, rwnn_cross_vals = rwnn.train_and_test()
    vanilla_obj_vals, vanilla_cross_vals = vanilla.train_and_test(500)

    # pass results to graphic visualizer (training/testing plot)
    # plot_train_test(train_results, test_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)