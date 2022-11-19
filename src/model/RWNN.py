'''Class for the RWNN portion of model'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class RWNN(nn.Module):
    def __init__(self, activation_function:str, case:int, verbose:bool=True):
        super().__init__()

    def forward(self):
        pass