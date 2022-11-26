'''Functions for plotting training and testing results'''

import matplotlib.pyplot as plt

def plot_train_test(rwnn_obj_vals, rwnn_cross_vals, vanilla_obj_vals, vanilla_cross_vals, res_path):
    '''Plots the training and testing results'''
    
    assert len(rwnn_obj_vals) == len(rwnn_cross_vals), 'Length mismatch between RWNN loss curves'
    num_epochs = len(rwnn_obj_vals[-1])

    # Plot saved in results folder
    plt.plot(range(num_epochs), rwnn_obj_vals[-1], label= "Training", color="blue")
    plt.plot(range(num_epochs), rwnn_cross_vals[-1], label= "Testing", color= "green")
    plt.legend()
    plt.title('RWNN Model Loss for Final Iteration')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + 'rwnn_loss_fig.jpeg')
    plt.close()
    
    assert len(vanilla_obj_vals) == len(vanilla_cross_vals), 'Length mismatch between vanilla loss curves'
    num_epochs = len(vanilla_obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), vanilla_obj_vals, label= "Training", color="blue")
    plt.plot(range(num_epochs), vanilla_cross_vals, label= "Testing", color= "green")
    plt.legend()
    plt.title('Vanilla Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + 'van_loss_fig.pdf')
    plt.close()