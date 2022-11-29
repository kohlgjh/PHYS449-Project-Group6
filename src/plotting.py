'''Functions for plotting training and testing results'''

import matplotlib.pyplot as plt

def plot_train_test(rwnn_obj_vals, rwnn_cross_vals, vanilla_obj_vals, vanilla_cross_vals, res_path, case):
    '''Plots the training and testing results'''
    
    # change format of RWNN results to plot sequential iterations
    rwnn_obj_flat = [item for sublist in rwnn_obj_vals for item in sublist]
    rwnn_cross_flat =  [item for sublist in rwnn_cross_vals for item in sublist]
    
    assert len(rwnn_obj_flat) == len(rwnn_cross_flat), 'Length mismatch between loss curves'
    num_epochs = len(rwnn_obj_flat)

    # Plot saved in results folder
    plt.plot(range(num_epochs), rwnn_obj_flat, label= "RWNN Training", color="blue")
    plt.plot(range(num_epochs), rwnn_cross_flat, label= " RWNN Testing", color= "green")
    plt.plot(range(num_epochs), vanilla_obj_vals, label= "Vanilla Training", color="red")
    plt.plot(range(num_epochs), vanilla_cross_vals, label= " Vanilla Testing", color= "darkviolet")
    plt.legend()
    plt.title(f'Loss of RWNN and Vanilla Models\n Case {case}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + 'loss_fig.pdf')
    plt.close()
