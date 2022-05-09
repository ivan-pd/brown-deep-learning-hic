import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import os

from model import HiCPlus
from run import get_data, get_baseline_data, test

def plot_pearson(data, title):
    
    x = np.arange(len(data))
    y = data

    p = sns.lineplot(x, y)

    p.set_xlabel("Batch Number")
    p.set_ylabel("Pearson Correlation")
    p.set_title(title)
    plt.show()


def loss_vs_epoch():
    filename = './results/loss_per_epoch.txt'
    with open(filename) as file:
        lines = file.readlines()
        lines = [float(line.rstrip()) for line in lines]

    x = np.arange(len(lines))
    y = lines

    p = sns.lineplot(x, y)

    p.set_xlabel("Epoch")
    p.set_ylabel("Mean Squared Error Loss")
    title = "Model Training Loss per Epoch"

    p.set_title(title, pad=20)

    plt.show()


def model_accuracy(model, inputs, targets):
    accuracy = []
    num_batches = math.floor(inputs.shape[0] / model.batch_size)

    for i in range(num_batches):
        start = i * model.batch_size
        input_batch = inputs[start : start + model.batch_size]
        target_batch = targets[start : start + model.batch_size]
        predictions = model.call(input_batch)
        accuracy.append(model.accuracy(predictions, target_batch))

    graph_title = "Base Pearson Correlation"
    plot_pearson(accuracy, graph_title)


def correlation(predictions, targets):
    predictions_flat = np.reshape(predictions, [250, -1])
    targets_flat = np.reshape(targets, [250, -1])

    coefficents = []
    for i in range(predictions_flat.shape[0]):
        x = predictions_flat[i]
        y = targets_flat[i]
        r = np.corrcoef(x, y)
        coefficents.append(r[0][1])

    return np.average(coefficents)

def get_accuracies():
    # Step 0: Load data
    test_downsample_16 = './synthetic/c40_s28/16_downsampling/test.npz'
    test_inputs, test_targets, test_inds = get_data(test_downsample_16)

    # Step 1: Load model weights at 200 epochs
    model = HiCPlus()
    weights_path = os.path.join("model_ckpts", "model_200")
    model.load_weights('./model_ckpts/model_200')

    # Step 2: Get Model Test Accuracy 
    model_acc = test(model, test_inputs, test_targets)

    # Step 3: Get baseline model accuracy
    base_inputs, base_targets, test_inds = get_baseline_data(test_downsample_16)

    accuracy = []
    num_batches = math.floor(base_inputs.shape[0] / model.batch_size)
    for i in range(num_batches):
        start = i * model.batch_size
        input_batch = base_inputs[start : start + model.batch_size]
        target_batch = base_targets[start : start + model.batch_size]
        accuracy.append(correlation(input_batch, target_batch))

    # Step 4: Plot baseline and Model Accuracy for comparison
    data = []
    for i in range(len(accuracy)):
        base_data_pt = [i, 'Base', accuracy[i]]
        model_data_pt = [i, 'HiCPlus', model_acc[i]]

        data.append(base_data_pt)
        data.append(model_data_pt)

    # creating a list of column names
    column_values = ['batch_number', 'Model', 'accuracy']
    
    # creating the dataframe
    df = pd.DataFrame(data = data, columns = column_values)
    
    sns.lineplot(x = "batch_number", y = "accuracy", data = df, hue = "Model")
 
    plt.title("Average Batch Accuracy of Model and Base", fontsize = 14) # for title
    plt.xlabel("Batch Number", fontsize = 10) # label for x-axis
    plt.ylabel("Accuracy (MSE)", fontsize = 10) # label for y-axis
    plt.show()
    

def main():
    # loss_vs_epoch()
    get_accuracies()

    
   

if __name__ == '__main__':
    main()
