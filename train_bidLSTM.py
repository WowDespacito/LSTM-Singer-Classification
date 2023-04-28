#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from GenreFeatureData import (
    GenreFeatureData,
)  # local python class with Audio feature extraction (librosa)
from myModule import bidLSTM

def main():
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print("\nTraining on GPU")
    else:
        print("\nNo GPU, training on CPU")

    device = torch.device('cuda' if train_on_gpu else 'cpu')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    genre_features = GenreFeatureData()

    # if all of the preprocessed files do not exist, regenerate them all for self-consistency
    if (
            os.path.isfile(genre_features.train_X_preprocessed_data)
            and os.path.isfile(genre_features.train_Y_preprocessed_data)
            and os.path.isfile(genre_features.dev_X_preprocessed_data)
            and os.path.isfile(genre_features.dev_Y_preprocessed_data)
    ):
        print("Preprocessed files exist, deserializing npy files")
        genre_features.load_deserialize_data()
    else:
        print("Preprocessing raw audio files")
        genre_features.load_preprocess_data()

    train_X = torch.from_numpy(genre_features.train_X).type(torch.Tensor).to(device)
    dev_X = torch.from_numpy(genre_features.dev_X).type(torch.Tensor).to(device)

    # Targets is a long tensor of size (N,) which tells the true class of the sample.
    train_Y = torch.from_numpy(genre_features.train_Y).type(torch.LongTensor).to(device)
    dev_Y = torch.from_numpy(genre_features.dev_Y).type(torch.LongTensor).to(device)

    # Convert {training, test} torch.Tensors
    print("Training X shape: " + str(genre_features.train_X.shape))
    print("Training Y shape: " + str(genre_features.train_Y.shape))
    print("Validation X shape: " + str(genre_features.dev_X.shape))
    print("Validation Y shape: " + str(genre_features.dev_Y.shape))

    batch_size = 100  # num of training examples per minibatch
    num_epochs = 1000

    # Define model
    print("Build LSTM RNN model ...")
    model = bidLSTM(
        input_dim=33, hidden_dim=256, batch_size=batch_size, output_dim=5, num_layers=2, bidirectional=True
    ).to(device)
    # state_dict = torch.load('./weights/model_parameter.pkl')
    # model.load_state_dict(state_dict)



    loss_function = nn.NLLLoss()  # expects ouputs from LogSoftmax

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # To keep LSTM stateful between batches, you can set stateful = True, which is not suggested for training
    stateful = False

    # all training data (epoch) / batch_size == num_batches (12)
    num_batches = int(train_X.shape[0] / batch_size)
    num_dev_batches = int(dev_X.shape[0] / batch_size)

    val_loss_list, val_accuracy_list, epoch_list = [], [], []
    tra_loss_list, tra_accuracy_list, Epoch_list = [], [], []

    print("Training ...")
    for epoch in range(num_epochs):

        train_running_loss, train_acc = 0.0, 0.0

        # Init hidden state - if you don't want a stateful LSTM (between epochs)
        hidden_state = None
        for i in range(num_batches):

            # zero out gradient, so they don't accumulate btw batches
            model.zero_grad()

            # train_X shape: (total # of training examples, sequence_length, input_dim)
            # train_Y shape: (total # of training examples, # output classes)
            #
            # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
            # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
            X_local_minibatch, y_local_minibatch = (
                train_X[i * batch_size: (i + 1) * batch_size, ],
                train_Y[i * batch_size: (i + 1) * batch_size, ],
            )

            # Reshape input & targets to "match" what the loss_function wants
            X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

            # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
            y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

            y_pred, hidden_state = model(X_local_minibatch, hidden_state)  # forward pass

            # Stateful = False for training. Do we go Stateful = True during inference/prediction time?
            if not stateful:
                hidden_state = None
            else:
                h_0, c_0 = hidden_state
                h_0.detach_(), c_0.detach_()
                hidden_state = (h_0, c_0)

            loss = loss_function(y_pred, y_local_minibatch)  # compute loss
            loss.backward()  # backward pass
            optimizer.step()  # parameter update

            train_running_loss += loss.detach().item()  # unpacks the tensor into a scalar value
            train_acc += model.get_accuracy(y_pred, y_local_minibatch)

        print(
            "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
            % (epoch, train_running_loss / num_batches, train_acc / num_batches)
        )
        Epoch_list.append(epoch)
        tra_loss_list.append(train_running_loss / num_batches)
        tra_accuracy_list.append(train_acc / num_batches)
        if epoch % 10 == 0:
            print("Validation ...")  # should this be done every N=10 epochs
            val_running_loss, val_acc = 0.0, 0.0

            # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
            with torch.no_grad():
                model.eval()

                hidden_state = None
                for i in range(num_dev_batches):
                    X_local_validation_minibatch, y_local_validation_minibatch = (
                        dev_X[i * batch_size: (i + 1) * batch_size, ],
                        dev_Y[i * batch_size: (i + 1) * batch_size, ],
                    )
                    # print(type(X_local_validation_minibatch))
                    X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                    # print(X_local_minibatch.shape)
                    y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                    y_pred, hidden_state = model(X_local_minibatch, hidden_state)
                    # print(y_pred.shape)
                    if not stateful:
                        hidden_state = None

                    val_loss = loss_function(y_pred, y_local_minibatch)

                    val_running_loss += (
                        val_loss.detach().item()
                    )  # unpacks the tensor into a scalar value
                    val_acc += model.get_accuracy(y_pred, y_local_minibatch)

                model.train()  # reset to train mode after iterationg through validation data
                print(
                    "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                    % (
                        epoch,
                        train_running_loss / num_batches,
                        train_acc / num_batches,
                        val_running_loss / num_dev_batches,
                        val_acc / num_dev_batches,
                    )
                )

            epoch_list.append(epoch)
            val_accuracy_list.append(val_acc / num_dev_batches)
            val_loss_list.append(val_running_loss / num_dev_batches)

    # visualization loss
    fig, ax = plt.subplots(1, 2)
    ax1 = ax[0]
    ax2 = ax[1]
    ax1.plot(Epoch_list, tra_loss_list, label='train')
    ax1.plot(epoch_list, val_loss_list, label='val')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("bidLSTM: Loss")
    # visualization accuracy
    ax2.plot(Epoch_list, tra_accuracy_list, label='train')
    ax2.plot(epoch_list, val_accuracy_list, label='val')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("bidLSTM: Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bidResult")

    torch.save(model.state_dict(), "./weights/bidlstm_parameter.pkl")

if __name__ == "__main__":
    main()
