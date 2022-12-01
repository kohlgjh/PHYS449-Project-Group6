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
                 input_size:int, hidden_size: int, subset_size:int, device:str, seed:int) -> None:
        # save params as attributes
        self.epochs = epochs
        self.iterations = iterations
        self.verbose = verbose
        self.subset_size = subset_size

        # data
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self._generate_subsets(seed)

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

    def _generate_subsets(self, seed):
        '''Method to generate subsets of training data, and return them'''  
        num_samples = self.train_input.shape[0]
        num_feature = self.train_input.shape[1]
        np.random.seed(seed)
        random_ints = np.random.randint(0, num_samples, (self.iterations, self.subset_size))

        input_subsets = np.empty((self.iterations, self.subset_size, num_feature))
        target_subsets = np.empty((self.iterations, self.subset_size, 3))

        for i in range(self.iterations):
            input_subsets[i] = self.train_input[random_ints[i]]
            target_subsets[i] = self.train_target[random_ints[i]]

        self.train_input = input_subsets.copy()
        self.train_target = target_subsets.copy()

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
        all_train_acc = np.empty((self.iterations, self.epochs,3), dtype=list)
        all_test_acc = np.empty((self.iterations, self.epochs,3), dtype=list)

        for iteration in range(self.iterations):

            iter_obj_vals = []
            iter_cross_vals = []
            iter_train_vals = []
            iter_test_vals = []

            for epoch in range(self.epochs):

                obj_val = self.loss_fct(self.model.forward(self.train_input[iteration]), self.train_target[iteration])

                self.optimizer.zero_grad()
                obj_val.backward()
                self.optimizer.step()
                iter_obj_vals.append(obj_val.item())
                train_acc = self._accuracy(self.train_input[iteration], self.train_target[iteration])
                iter_train_vals.append(train_acc)
                

                # test as training goes on
                with torch.no_grad():
                    cross_val = self.loss_fct(self.model.forward(self.test_input), self.test_target)
                    iter_cross_vals.append(cross_val.item())
                    test_acc = self._accuracy(self.test_input, self.test_target)
                    iter_test_vals.append(test_acc)

                if self.verbose:
                    if epoch == 0:
                        print(f"Iteration: {iteration+1}/{self.iterations}")

            # store current iteration's cross and obj vals
            all_obj_vals[iteration] = iter_obj_vals
            all_cross_vals[iteration] = iter_cross_vals
            all_train_acc[iteration] = train_acc
            all_test_acc[iteration] = test_acc
            

            # console output to track training
            if self.verbose:
                print(f"End of iteration: \t Training Loss: {obj_val.item():.3f} \t Test Loss: {cross_val.item():.3f}")
                print(f"\t\t\t Training Accuracy: Un: {train_acc[0]:.1f}%  Ps: {train_acc[1]:.1f}%  Mes: {train_acc[2]:.1f}%")
                print(f"\t\t\t Test Accuracy:     Un: {test_acc[0]:.1f}%  Ps: {test_acc[1]:.1f}%  Mes: {test_acc[2]:.1f}%\n")


        if self.verbose:
            print("End of RWNN training...\n")
        return all_obj_vals, all_cross_vals, all_train_acc, all_test_acc
            
