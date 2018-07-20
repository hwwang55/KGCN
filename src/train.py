import tensorflow as tf
import numpy as np
from model import KGCN


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def train(args, data, show_loss):
    n_user, n_entity, n_relation = data[0], data[1], data[2]
    train_data, eval_data, test_data = data[3], data[4], data[5]
    adj_entity, adj_relation = data[6], data[7]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start + args.batch_size <= train_data.shape[0]:  # skip the last batch if its size < batch size
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # evaluation
            train_auc, train_f1 = evaluation(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = evaluation(sess, model, eval_data, args.batch_size)
            test_auc, test_f1 = evaluation(sess, model, test_data, args.batch_size)

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))


def evaluation(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list))
