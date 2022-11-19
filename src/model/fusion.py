'''Class structure for fusion network'''

from GAN import GAN
from RWNN import RWNN
import numpy as np

class FusionNet():
    def __init__(self, activation_function:str, case:int, verbose:bool=True) -> None:
        '''
        Class structure for fusion network. 
        Combines RWNN and GAN networks
        
        Parameters
        ---
        activation_function: string
            "sigmoid" or "SBAF" to select version with the corresponding activation function
        case: int
            integer number 1-8 of feature case to use (affects size of input layer)
        verbose: bool
            True to turn on high verbosity
        '''
        self.RWNN = RWNN(activation_function, case, verbose=verbose)
        self.GAN = GAN(activation_function, case, verbose=verbose)

    def train(self) -> np.ndarray:
        '''Trains both RWNN and GAN models'''
        self._train_RWNN() # train the RWNN portion of the model
        self._train_GAN() # train teh GAN portino of the model

        results = np.array([])
        return results        

    def test(self) -> np.ndarray:
        '''
        Test the fusion model
        
        Returns:
        ---
        numpy array of the results
        '''
        results = np.array([])
        return results

    def _train_RWNN(self):
        '''Trains the RWNN model'''

    def _train_GAN(self):
        '''Trains the GAN model'''