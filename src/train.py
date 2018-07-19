import tensorflow as tf
import numpy as np


def train(args, data):
    n_user, n_entity, n_relation = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    adj_entity, adj_relation = data[6], data[7]
