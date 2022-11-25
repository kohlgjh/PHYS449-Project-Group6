'''Class for the creating and training RWNN model'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model
import numpy as np

class RWNN():
    '''
    Class structure to handle the RWNN training of the base model.

    Params
    ---
    data: numpy arrays
        4 arrays of training input, trainig target, test input, and test test target
    epochs: int
        number of training epochs per iteration
    iterations: int
        number of iterations in RWNN training, each contains "epochs" number of epochs
    learning_rate: float
        learning rate of the optimizer
    momentum: float
        momentum of the optimizer
    verbose: bool
        True/False, to turn on/off verbose output during training


    The number of samples in each iteration of training is given by the total number
    of samples divided by the number of iterations. Each of these sample subsets are
    trained for the given number of epohcs, and the weights from the previous iteration
    carry over to the next iteration.
    '''
    def __init__(self, data, epochs:int, iterations:int, learning_rate: float, momentum:float, verbose:bool,
                 input_size:int, hidden_size: int) -> None:
        # save params as attributes
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose

        # data
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self._generate_subsets()

        # intitialize model
        self.model = model(input_size, hidden_size)

        # optimizer and loss function
        self.loss_fct = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

    def _generate_subsets(self):
        '''Method to generate subsets of training data, and return them'''  
        num_samples = self.train_input.shape[0]
        num_feature = self.train_input.shape[1]
        subset_size = int(num_samples/self.iterations)
        trim = num_samples - self.iterations*subset_size # amount to trim to reshape properly

        self.train_input = np.reshape(self.train_input[:-trim], (self.iterations, subset_size, num_feature))
        self.train_target = np.reshape(self.train_target[:-trim], (self.iterations, subset_size, 3))

    def train_and_test(self):
        '''Trains and tests the RWNN model'''

        for iteration in range(self.iterations):

            print("training iteration n...") # placeholder for now

            # testing results of this iteration...

            # repeat...

            
