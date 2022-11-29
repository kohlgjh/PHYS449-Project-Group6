'''Class for the creating and training RWNN model'''
import torch
import torch.nn as nn
from src.model.model import BaseModel
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
    input_size: int
        number of input neurons
    hidden_size: int
        number of neurons in the hidden layer
    subset_size: int
        number of samples in each subset
    device: str
        "cpu" or "cuda" to use the corresponding device 


    The number of samples in each iteration of training is given by the total number
    of samples divided by the number of iterations. Each of these sample subsets are
    trained for the given number of epohcs, and the weights from the previous iteration
    carry over to the next iteration.
    '''
    def __init__(self, data, epochs:int, iterations:int, learning_rate: float, momentum:float, verbose:bool,
                 input_size:int, hidden_size: int, subset_size:int, device:str) -> None:
        # save params as attributes
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose
        self.subset_size = subset_size

        # data
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self._generate_subsets()

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

    def _generate_subsets(self):
        '''Method to generate subsets of training data, and return them'''  
        num_samples = self.train_input.shape[0]
        num_feature = self.train_input.shape[1]
        random_ints = np.random.randint(0, num_samples, (self.iterations, self.subset_size))

        input_subsets = np.empty((self.iterations, self.subset_size, num_feature))
        target_subsets = np.empty((self.iterations, self.subset_size, 3))

        for i in range(self.iterations):
            input_subsets[i] = self.train_input[random_ints[i]]
            target_subsets[i] = self.train_target[random_ints[i]]

        self.train_input = input_subsets.copy()
        self.train_target = target_subsets.copy()

    def _accuracy(self, input, target) -> float:
        '''Returns accuracy evaluation as a percentage'''
        model_results = self.model.forward(input).cpu().detach().numpy()
        targets = target.cpu().detach().numpy().astype(int)

        # make highest val 1 and the rest 0
        results_processed = np.zeros_like(model_results)
        results_processed[np.arange(model_results.shape[0]), model_results.argmax(1)] = 1
        results_processed = results_processed.astype(int)

        # compare to the targets
        correct = 0
        for prediction, answer in zip(results_processed, targets):
            if np.array_equal(prediction, answer): correct += 1

        return (correct/targets.shape[0])*100



    def train_and_test(self) -> np.ndarray:
        '''
        Trains and tests the RWNN model
        
        Returns arrays of all_objective_values, and all_cross_vals of shape:
        (num_iterations, num_epochs)
        '''
        if self.verbose:
            print("Beginning RWNN training...")

        self.model.reset()

        all_obj_vals = np.empty((self.iterations, self.epochs), dtype=list)
        all_cross_vals = np.empty((self.iterations, self.epochs), dtype=list)

        for iteration in range(self.iterations):

            iter_obj_vals = []
            iter_cross_vals = []

            for epoch in range(self.epochs):

                obj_val = self.loss_fct(self.model.forward(self.train_input[iteration]), self.train_target[iteration])

                self.optimizer.zero_grad()
                obj_val.backward()
                self.optimizer.step()
                iter_obj_vals.append(obj_val.item())

                # test as training goes on
                with torch.no_grad():
                    cross_val = self.loss_fct(self.model.forward(self.test_input), self.test_target)
                    iter_cross_vals.append(cross_val.item())

                if self.verbose:
                    if epoch == 0:
                        train_accuracy = self._accuracy(self.train_input[iteration], self.train_target[iteration])
                        test_accuracy = self._accuracy(self.test_input, self.test_target)
                        print(f"Iteration: {iteration+1}/{self.iterations}")
                        print(f"Start of iteration: \t Training Loss: {obj_val.item():.3f}  Training Accuracy: {train_accuracy:.1f}% \t Test Loss: {cross_val.item():.3f} Test Accuracy: {test_accuracy:.1f}%")


            # store current iteration's cross and obj vals
            all_obj_vals[iteration] = iter_obj_vals
            all_cross_vals[iteration] = iter_cross_vals

            # console output to track training
            if self.verbose:
                train_accuracy = self._accuracy(self.train_input[iteration], self.train_target[iteration])
                test_accuracy = self._accuracy(self.test_input, self.test_target)
                print(f"End of iteration: \t Training Loss: {obj_val.item():.3f}  Training Accuracy: {train_accuracy:.1f}% \t Test Loss: {cross_val.item():.3f}  Test Accuracy: {test_accuracy:.1f}%\n")

        if self.verbose:
            print("End of RWNN training...\n")
        return all_obj_vals, all_cross_vals
            
