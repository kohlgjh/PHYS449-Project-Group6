'''Functions to upsample/downsample, and generate training and testing data'''
import pandas as pd
import numpy as np
import os

def generate_train_and_test(case_num, seed=1234, num_samples=680):
    '''
    Reads in CSV for specified case and returns 80/20 split of train/test:
    
    Params
    ---
    case_num: int
        1-8: feature case to use
    seed: int
        seed to feed to random number generator for bootstrapping
    num_samples: int
        number of samples of each kind of planet. Final size = 3*num_samples

    Returns
    ---
    train_input:
        2D numpy array of input data
    train_target
        2D numpy array of corresponding targets: 0 = [1, 0, 0], 1 = [0, 1, 0], 2 = [0, 0, 1]
    test_input:
        likewise, but for testing
    test_target:
        likewise, but for testing
    '''

    df = pd.read_csv(os.path.join(os.getcwd(), f'data\\processed\\PHL-EC-Case{int(case_num)}.csv'))

    # split data into 3 dataframes based on habitability label
    df_0 = df[df["hab_lbl"] == 0]
    df_1 = df[df["hab_lbl"] == 1]
    df_2 = df[df["hab_lbl"] == 2]

    # generate random indices for selectign samples
    np.random.seed(seed)
    rand_num_0 = np.random.randint(0, df_0.shape[0], size = num_samples)
    rand_num_1 = np.random.randint(0, df_1.shape[0], size = num_samples)
    rand_num_2 = np.random.randint(0, df_2.shape[0], size = num_samples)

    # convert to numpy arrays and sample
    agg_0 = df_0.to_numpy()[rand_num_0]
    agg_1 = df_1.to_numpy()[rand_num_1]
    agg_2 = df_2.to_numpy()[rand_num_2]

    # doing 80/20 train/test split we concatenate
    train = np.concatenate((agg_0[0:int(num_samples*0.8), :], agg_1[0:int(num_samples*0.8), :], agg_2[0:int(num_samples*0.8), :]))
    test = np.concatenate((agg_0[int(num_samples*0.8):, :], agg_1[int(num_samples*0.8):, :], agg_2[int(num_samples*0.8):, :]))

    # separating inputs and targets
    train_input, train_target1D = train[:, 1:], train[:, 0]
    test_input, test_target1D = test[:, 1:], test[:, 0]

    # turning targets from 1 -> [0, 1, 0], 2 -> [0, 0, 1], etc.
    train_target = np.empty((len(train_target1D), 3), dtype=int)
    train_target[np.where(train_target1D == 0), :] = [1, 0, 0]
    train_target[np.where(train_target1D == 1), :] = [0, 1, 0]
    train_target[np.where(train_target1D == 2), :] = [0, 0, 1]

    test_target = np.empty((len(test_target1D), 3), dtype=int)
    test_target[np.where(test_target1D == 0), :] = [1, 0, 0]
    test_target[np.where(test_target1D == 1), :] = [0, 1, 0]
    test_target[np.where(test_target1D == 2), :] = [0, 0, 1]

    return train_input, train_target, test_input, test_target