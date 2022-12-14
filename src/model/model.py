import torch
import torch.nn as nn

class BaseModel(nn.Module):
    '''
    Base model used by both vanilla and RWNN networks.

    Params
    ---
    input_size: int
        number of input neurons corresdponding to number of features
    hidden_size: int
        number of neurons in the hidden layer
    '''
    def __init__(self, input_size:int, hidden_size:int) -> None:
        super().__init__()
        
        # define layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 3) # every case has three output neurons
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        '''Takes input data and returns softmaxed guess of exoplanet type'''
        x = self.input_layer(x)
        x = torch.sigmoid(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x

    def reset(self):
        '''Resets params of the model layers'''
        self.input_layer.reset_parameters()
        self.output_layer.reset_parameters()
