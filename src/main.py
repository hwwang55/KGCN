import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use,')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=0.0001, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
args = parser.parse_args()

show_loss = False
data = load_data(args)
train(args, data, show_loss)
