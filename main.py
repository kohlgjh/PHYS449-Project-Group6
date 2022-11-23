'''Main entry point into code'''
import argparse, json, sys
sys.path.append('src')
from src.model.RWNN import RWNN
from src.visualizer.plotting import plot_train_test
from src.data_processing.process_data import generate_train_and_test
import numpy as np
import torch.nn as nn

def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, help="integer 1-8 of what feature set to use")
    parser.add_argument('-v', "--verbose", default="True", help="True/False for high/low verbosity")
    parser.add_argument("--seed", default=1234, help="random seed")
    parser.add_argument("--params", default='input/params.json', help="JSON filepath containing hyperparameters")
    return parser.parse_args()

def main(args):
    '''Main code block'''
    verbose = True if args.verbose == "True" else False
    case = int(args.case)
    
    np.random.seed(args.seed)
    
    # hyperparameters from json file
    with open(args.params) as paramfile:
        param = json.load(paramfile)

    # generate datasets
    train_input, train_target, test_input, test_target = generate_train_and_test(case)

    # run RWNN model
    # rwnn = RWNN(case, verbose)
    loss_fct = nn.MSELoss()
    optimizer = nn.Adam()
    
    # train_results = rwnn.train()
    # test_results = rwnn.test()

    # pass results to graphic visualizer (training/testing plot)
    # plot_train_test(train_results, test_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)