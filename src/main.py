import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
args = parser.parse_args()

data = load_data(args)
train(args, data)
