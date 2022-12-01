'''Functions for plotting training and testing results'''

import matplotlib.pyplot as plt

def plot_train_test(rwnn_obj_vals, rwnn_cross_vals, rwnn_train_acc, rwnn_test_acc, vanilla_obj_vals, vanilla_cross_vals, vanilla_train_acc, vanilla_test_acc, res_path, case):
    '''Plots the training and testing results'''
    
    # change format of RWNN results to plot sequential iterations
    rwnn_obj_flat = [item for sublist in rwnn_obj_vals for item in sublist]
    rwnn_cross_flat =  [item for sublist in rwnn_cross_vals for item in sublist]
    rwnn_train_flat = [item for sublist in rwnn_train_acc for item in sublist]
    rwnn_test_flat = [item for sublist in rwnn_test_acc for item in sublist]
    
    # check that the lengths match for plotting
    assert len(rwnn_obj_flat) == len(rwnn_cross_flat), 'Length mismatch between loss curves'
    num_epochs = len(rwnn_obj_flat)

    # Plot the loss and save in the appropriate folder
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
    
    # check that the lengths match for plotting
    num_epochs_rwnn = len(rwnn_test_flat)
    num_epochs_van = len(vanilla_test_acc)

    # Plot the loss and save in the appropriate folder
    plt.plot(range(num_epochs_rwnn), [item[0] for item in rwnn_train_flat], label= "RWNN Training - Unhabitable")
    plt.plot(range(num_epochs_rwnn), [item[1] for item in rwnn_train_flat], label= "RWNN Training - Meso")
    plt.plot(range(num_epochs_rwnn), [item[2] for item in rwnn_train_flat], label= "RWNN Training - Psychro")
    plt.plot(range(num_epochs_rwnn), [item[0] for item in rwnn_test_flat], label= "RWNN Testing - Unhabitable")
    plt.plot(range(num_epochs_rwnn), [item[1] for item in rwnn_test_flat], label= "RWNN Testing - Meso")
    plt.plot(range(num_epochs_rwnn), [item[2] for item in rwnn_test_flat], label= "RWNN Testing - Psychro")
    plt.legend()
    plt.title(f'Accuracy of RWNN Model\n Case {case}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(res_path + 'rwnn_accuracy_fig.pdf')
    plt.close()
    
    plt.plot(range(num_epochs_van), [item[0] for item in vanilla_train_acc], label= "Vanilla Training - Unhabitable")
    plt.plot(range(num_epochs_van), [item[1] for item in vanilla_train_acc], label= "Vanilla Training - Meso")
    plt.plot(range(num_epochs_van), [item[2] for item in vanilla_train_acc], label= "Vanilla Training - Psychro")
    plt.plot(range(num_epochs_van), [item[0] for item in vanilla_test_acc], label= "Vanilla Testing - Unhabitable")
    plt.plot(range(num_epochs_van), [item[1] for item in vanilla_test_acc], label= "Vanilla Testing - Meso")
    plt.plot(range(num_epochs_van), [item[2] for item in vanilla_test_acc], label= "Vanilla Testing - Psychro")
    plt.legend()
    plt.title(f'Accuracy of Vanilla Model\n Case {case}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(res_path + 'van_accuracy_fig.pdf')
    plt.close()