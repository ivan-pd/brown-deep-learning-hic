import tensorflow as tf
import numpy as np
import math
import os

from model import HiCPlus
from numpy import savetxt
# from graphs import plot_pearson

def train(model, train_inputs, train_targets):

    # Shuffling inputs to prevent overfitting 
    num_examples = train_inputs.shape[0] 
    shuffle_indicies = tf.random.shuffle(tf.range(num_examples))
    train_inputs = tf.gather(train_inputs, shuffle_indicies)
    train_targets = tf.gather(train_targets, shuffle_indicies)

    accuracy = []
    losses =  []
    num_batches = math.floor(num_examples / model.batch_size)
    for i in range(num_batches):
        start = i * model.batch_size
        
        input_batch = train_inputs[start : start + model.batch_size]
        target_batch = train_targets[start : start + model.batch_size]

        with tf.GradientTape() as tape: 
            predictions = model.call(input_batch)
            loss = model.loss_function(predictions, target_batch)
            losses.append(loss)
            accuracy.append(model.accuracy(predictions, target_batch))

        train_vars = model.trainable_variables
        gradients = tape.gradient(loss, train_vars)
        model.opt.apply_gradients(zip(gradients, train_vars))

    return accuracy, losses


def test(model, test_input, test_labels):
    accuracy = []
    num_batches = math.floor(test_input.shape[0] / model.batch_size)
    for i in range(num_batches):
        start = i * model.batch_size
        input_batch = test_input[start : start + model.batch_size]
        target_batch = test_labels[start : start + model.batch_size]
        predictions = model.call(input_batch)
        accuracy.append(model.accuracy(predictions, target_batch))

    return accuracy


def get_data(file):
    with np.load(file) as hic_data:
        inputs = hic_data['data']
        targets = hic_data['target']
        inds = hic_data['inds']

    # Reshaping targets to match 28x28 output of HiCPlus Model
    shape = (targets.shape[0], 1, 28, 28)
    targets_reshaped = np.zeros(shape)
    for i in range(targets.shape[0]):
        targets_reshaped[i] = targets[i, 0, 6:34, 6:34]

    # Squeezing out extra dimentsion  to convert from (N, C, H, W) to (N, H, W, C) format
    inputs_reshaped = np.squeeze(inputs)
    targets_reshaped = np.squeeze(targets_reshaped)

    # Adding Channel
    inputs_reshaped = inputs_reshaped.reshape((inputs_reshaped.shape[0], 40, 40, 1))
    targets_reshaped = targets_reshaped.reshape((targets_reshaped.shape[0], 28, 28, 1))
   
    return inputs_reshaped, targets_reshaped, inds

def get_baseline_data(file):
    with np.load(file) as hic_data:
        inputs = hic_data['data']
        targets = hic_data['target']
        inds = hic_data['inds']

    shape = (targets.shape[0], 1, 28, 28)
    inputs_reshaped = np.zeros(shape)
    targets_reshaped = np.zeros(shape)
    for i in range(targets.shape[0]):
        inputs_reshaped[i] = inputs[i, 0, 6:34, 6:34]
        targets_reshaped[i] = targets[i, 0, 6:34, 6:34]
    
    inputs_reshaped = np.squeeze(inputs_reshaped)
    targets_reshaped = np.squeeze(targets_reshaped)

    inputs_reshaped = inputs_reshaped.reshape((inputs_reshaped.shape[0], 28, 28, 1))
    targets_reshaped = targets_reshaped.reshape((targets_reshaped.shape[0], 28, 28, 1))
   
    return inputs_reshaped, targets_reshaped, inds


# Following save and load model weights functions adapted from VAE Project

def save_model_weights(model, file_name):
    """
    Save trained CNN model weights to model_ckpts/

    Inputs:
    - model: Trained CNN model.
    - args: All arguments.
    """

    output_path = os.path.join(file_name)
    os.makedirs("model_ckpts", exist_ok=True)
    model.save_weights(output_path)
   

def main():
    
    train_downsample_16 = './synthetic/c40_s28/16_downsampling/train.npz'
    test_downsample_16 = './synthetic/c40_s28/16_downsampling/test.npz'

    train_inputs, train_targets, train_inds = get_data(train_downsample_16)
    test_inputs, test_targets, test_inds = get_data(test_downsample_16)

    model = HiCPlus()
    
    epochs = 250
    test_accuracy = []
    for i in range(epochs):
        print(f'Running Epoch: {i}')
        acc, loss = train(model, train_inputs, train_targets)

        # Checking model accuracy every 25 epochs
        if i % 25 == 0:
            print(f'Testing Model on Epoch {i}')
            curr_acc = test(model, test_inputs, test_targets)
            test_accuracy.append(curr_acc)

            file_name = 'acc_' + str(i) + '.csv'
            savetxt(file_name, test_accuracy, delimiter=',')

        # Saving Final Loss at Epoch 
        with open('loss.txt', "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)

            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write("\n")

            # Append text at the end of file
            final_loss = float(loss[-1])
            file_object.write(str(final_loss))
            
            file_object.close()      

        # Saving Model Weights
        if i % 100 == 0:
            print(f'Saving Model Weights on Epoch {i}')
            file_name = 'model_' + str(i)
            save_model_weights(model, file_name)

            
    test_accuracy = test(model, test_inputs, test_targets)


if __name__ == '__main__':
    main()

