'''Class for creating and training the vanilla model'''
import torch
import torch.nn as nn
from src.model.model import BaseModel
import numpy as np


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
    device: str
        "cpu" or "cuda" to use the corresponding device 
    '''
    def __init__(self, data, epochs:int, iterations:int, learning_rate: float, momentum:float, verbose:bool,
                 input_size:int, hidden_size: int, device:str) -> None:
        # save params as attributes
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose

        # data
        self.train_input, self.train_target, self.test_input, self.test_target = data

        # convert to tensors
        self.train_input = torch.from_numpy(self.train_input).type(torch.float).to(device)
        self.train_target = torch.from_numpy(self.train_target).type(torch.float).to(device)
        self.test_input = torch.from_numpy(self.test_input).type(torch.float).to(device)
        self.test_target = torch.from_numpy(self.test_target).type(torch.float).to(device)

        # intitialize model
        self.model = BaseModel(input_size, hidden_size).to(device)

        # optimizer and loss function
        self.loss_fct = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum)

    def _accuracy(self, input, target) -> float:
        '''
        Returns accuracy evaluation percentage as a tuple, for each planet type:
        unhabitable, psychro-, and meso-
        
        Params
        ---
        input: 
            2D array of input data to be put through model
        target: 
            2D array of corresponding targets for input data
        '''
        model_results = self.model.forward(input).cpu().detach().numpy()
        targets = target.cpu().detach().numpy().astype(int)

        # make highest val 1 and the rest 0
        results_processed = np.zeros_like(model_results)
        results_processed[np.arange(model_results.shape[0]), model_results.argmax(1)] = 1
        results_processed = results_processed.astype(int)

        #separate into types of planets
        where0 = np.where((targets == (1, 0, 0)).all(axis=1))[0]
        where1 = np.where((targets == (0, 1, 0)).all(axis=1))[0]
        where2 = np.where((targets == (0, 0, 1)).all(axis=1))[0]

        # calculate num correct for each
        correct0, correct1, correct2 = 0, 0, 0

        for prediction, answer in zip(results_processed[where0], targets[where0]):
            if np.array_equal(prediction, answer): correct0 += 1

        for prediction, answer in zip(results_processed[where1], targets[where1]):
            if np.array_equal(prediction, answer): correct1 += 1

        for prediction, answer in zip(results_processed[where2], targets[where2]):
            if np.array_equal(prediction, answer): correct2 += 1

        # calcualte accuracy for each
        accuracy0 = correct0/len(where0)*100
        accuracy1 = correct1/len(where1)*100
        accuracy2 = correct2/len(where2)*100

        return accuracy0, accuracy1, accuracy2

    def train_and_test(self, display_epochs:int):
        '''
        Trains and tests the vanilla model.

        display_epochs: int
            will output to console every display_epochs number of epochs
        
        Returns arrays of objective_values and cross_vals of shape:
        (num_epochs)
        '''
        if self.verbose:
            print("Beginning vanilla training...")

        self.model.reset()

        obj_vals = []
        cross_vals = []

        for epoch in range(self.epochs * self.iterations):

            obj_val = self.loss_fct(self.model.forward(self.train_input), self.train_target)

            self.optimizer.zero_grad()
            obj_val.backward()
            self.optimizer.step()
            obj_vals.append(obj_val.item())

            # test as training goes on
            with torch.no_grad():
                cross_val = self.loss_fct(self.model.forward(self.test_input), self.test_target)
                cross_vals.append(cross_val.item())

            # console output to track training
            if (epoch+1) % display_epochs == 0:
                if self.verbose:
                    train_acc = self._accuracy(self.train_input, self.train_target)
                    test_acc = self._accuracy(self.test_input, self.test_target)
                    print(f"Epoch {epoch+1}/{self.epochs*self.iterations}: \t Training Loss: {obj_val.item():.3f} \t Test Loss: {cross_val.item():.3f}")
                    print(f"\t\t\t Training Accuracy: Un: {train_acc[0]:.1f}%  Ps: {train_acc[1]:.1f}%  Mes: {train_acc[2]:.1f}%")
                    print(f"\t\t\t Test Accuracy:     Un: {test_acc[0]:.1f}%  Ps: {test_acc[1]:.1f}%  Mes: {test_acc[2]:.1f}%\n")

        if self.verbose:
            train_acc = self._accuracy(self.train_input, self.train_target)
            test_acc = self._accuracy(self.test_input, self.test_target)
            print(f"Final Results: \t\t Training Loss: {obj_val.item():.3f} \t Test Loss: {cross_val.item():.3f}")
            print(f"\t\t\t Training Accuracy: Un: {train_acc[0]:.1f}%  Ps: {train_acc[1]:.1f}%  Mes: {train_acc[2]:.1f}%")
            print(f"\t\t\t Test Accuracy:     Un: {test_acc[0]:.1f}%  Ps: {test_acc[1]:.1f}%  Mes: {test_acc[2]:.1f}%\n")


        if self.verbose:
            print("End of vanilla training...")
        return obj_vals, cross_vals
