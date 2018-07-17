import argparse
import numpy as np
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
args = parser.parse_args()

data = load_data(args)
train(args, data)
