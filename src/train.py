import tensorflow as tf
import numpy as np


def train(args, data):
    user_num, entity_num, relation_num = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    adj_entity, adj_relation = data[6], data[7]
