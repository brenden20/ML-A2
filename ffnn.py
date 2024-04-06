import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.dropout_rate = 0.5
        
        # Define the first layer
        self.W1 = nn.Linear(input_dim, h)
        self.bn1 = nn.BatchNorm1d(h)  # Batch normalization layer
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # Define the second layer
        self.W2 = nn.Linear(h, h)
        self.bn2 = nn.BatchNorm1d(h)  # Batch normalization layer
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Define the third layer
        self.W3 = nn.Linear(h, h)
        self.bn3 = nn.BatchNorm1d(h)  # Batch normalization layer
        self.activation3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout_rate)

        # Output layer
        self.output_dim = 5  # Assuming we have 5 classes
        self.W4 = nn.Linear(h, self.output_dim)
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Pass through the first layer
        x = self.dropout1(self.activation1(self.bn1(self.W1(input_vector))))
        
        # Pass through the second layer
        x = self.dropout2(self.activation2(self.bn2(self.W2(x))))
        
        # Pass through the third layer
        x = self.dropout3(self.activation3(self.bn3(self.W3(x))))
        
        # Output layer
        x = self.softmax(self.W4(x))
        
        return x

# The rest of the helper functions remain the same as in your provided code

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and prepare data
    # Assume the load_data function and others are defined elsewhere in the script
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Training loop with validation
    # This portion would involve training the model as per the revised structure
    # Note that training and validation loops would need to account for the modified architecture
