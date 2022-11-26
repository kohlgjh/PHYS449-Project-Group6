'''Class for creating and training the vanilla model'''
import torch
import torch.nn as nn
from src.model.model import BaseModel


class Vanilla():
    '''
    Class structure to handle the vanilla training of the base model.
    Params match RWNN in order to have equivalent amount of training
    but using a basic technique to train isntead.

    Params
    ---
    data: numpy arrays
        4 arrays of training input, trainig target, test input, and test test target
    epochs: int
        number of training epochs per iteration (in RWNN)
    iterations: int
        number of iterations in RWNN training, each contains "epochs" number of epochs
    learning_rate: float
        learning rate of the optimizer
    momentum: float
        momentum of the optimizer
    verbose: bool
        True/False, to turn on/off verbose output during training
    '''
    def __init__(self, data, epochs:int, iterations:int, learning_rate: float, momentum:float, verbose:bool,
                 input_size:int, hidden_size: int) -> None:
        # save params as attributes
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose

        # data
        self.train_input, self.train_target, self.test_input, self.test_target = data

        # intitialize model
        self.model = BaseModel(input_size, hidden_size)

        # optimizer and loss function
        self.loss_fct = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

    def train_and_test(self):
        '''Trains and tests the vanilla model'''