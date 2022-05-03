import tensorflow as tf
import numpy as np
import math

from hicplus import HiCPlus

def train(model, train_inputs, train_targets):

    # Shuffling inputs maybe improves accuracy?
    num_examples = train_inputs.shape[0] 
    # shuffle_indicies = tf.random.shuffle(tf.range(num_examples))
    # train_inputs = tf.gather(train_inputs, shuffle_indicies)
    # train_targets = tf.gather(train_targets, shuffle_indicies)

    num_batches = math.floor(num_examples / model.batch_size)
    for i in range(num_batches):
        start = i * model.batch_size
        input_batch = train_inputs[start : start + model.batch_size]
        target_batch = train_targets[start : start + model.batch_size]

        with tf.GradientTape() as tape: 
            predictions = model.call(input_batch)
            loss = model.loss_function(predictions, target_batch)

        train_vars = model.trainable_variables
        gradients = tape.gradient(loss, train_vars)
        model.opt.apply_gradients(zip(gradients, train_vars))

    return None


def test(model, test_input, test_labels):
    losses = []
    num_batches = math.floor(test_input.shape[0] / model.batch_size)
    for i in range(num_batches):
        start = i * model.batch_size
        input_batch = test_input[start : start + model.batch_size]
        label_batch = test_labels[start : start + model.batch_size]
        predictions = model.call(input_batch)
        losses.append(model.loss_function(predictions, label_batch))

    return losses


def get_data(file):
    with np.load(file) as hic_data:
        inputs = hic_data['data']
        targets = hic_data['target']
        inds = hic_data['inds']

   
    # Reshaping targets to match 28x28 output of HiCPlus Model
    shape = (inputs.shape[0], 1, 28, 28)
    targets_reshaped = np.zeros(shape)
    for i in range(targets.shape[0]):
        targets_reshaped[i] = targets[i, 0, 6:34, 6:34]

    # Transpose data to convert from (N, C, H, W) to (N, H, W, C) format
    inputs = tf.transpose(inputs, [0, 2, 3, 1])
    targets_reshaped = tf.transpose(targets_reshaped, [0, 2, 3, 1])

    return inputs, targets_reshaped, inds

def main():
    # inmodel = args.model
    # hicfile = args.inputfile
    # outname = args.outputfile

    # inputs, inds = open_data(hicfile)
    # input1 = inputs[0]
    # input1 =  np.squeeze(input1)
    
    train_downsample_16 = './synthetic/c40_s28/16_downsampling/train.npz'
    test_downsample_16 = './synthetic/c40_s28/16_downsampling/test.npz'

    train_inputs, train_targets, train_inds = get_data(train_downsample_16)
    test_inputs, test_targets, test_inds = get_data(test_downsample_16)
    
    # print(train_inputs.shape)
    # print(train_targets.shape)

    model = HiCPlus()
    epochs = 1
    for i in range(epochs):
        print(f'Running Epoch: {i}')
        train(model, train_inputs, train_targets)
  
    losses = test(model, test_inputs, test_targets)
    print('First 10 loss values')
    print(losses[:10])
    print('Last 10 loss values')
    print(losses[-10:])



if __name__ == '__main__':
    main()

