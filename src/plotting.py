'''Functions for plotting training and testing results'''

import matplotlib.pyplot as plt

def plot_train_test(rwnn_obj_vals, rwnn_cross_vals, vanilla_obj_vals, vanilla_cross_vals, res_path):
    '''Plots the training and testing results'''
    
    assert len(rwnn_obj_vals) == len(vanilla_obj_vals), 'Length mismatch between training loss curves'
    num_epochs = len(rwnn_obj_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), rwnn_obj_vals, label= "RWNN", color="blue")
    plt.plot(range(num_epochs), vanilla_obj_vals, label= "Vanilla", color= "green")
    plt.legend()
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + 'train_loss_fig.pdf')
    plt.close()
    
    assert len(rwnn_cross_vals) == len(vanilla_cross_vals), 'Length mismatch between testing loss curves'
    num_epochs = len(rwnn_cross_vals)

    # Plot saved in results folder
    plt.plot(range(num_epochs), rwnn_cross_vals, label= "RWNN", color="blue")
    plt.plot(range(num_epochs), vanilla_cross_vals, label= "Vanilla", color= "green")
    plt.legend()
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(res_path + 'test_loss_fig.pdf')
    plt.close()