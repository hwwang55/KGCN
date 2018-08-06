import networkx as nx
import numpy as np
import argparse


if __name__ == '__main__':
    np.random.seed(555)
    NUM = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='music')
    args = parser.parse_args()
    DATASET = args.d

    kg_np = np.load('../data/' + DATASET + '/kg_final.npy')
    kg = nx.Graph()
    kg.add_edges_from([(triple[0], triple[2]) for triple in kg_np])  # construct knowledge graph

    rating_np = np.load('../data/' + DATASET + '/ratings_final.npy')
    item_history = dict()
    item_set = set()
    for record in rating_np:
        user = record[0]
        item = record[1]
        rating = record[2]
        if rating == 1:
            if item not in item_history:
                item_history[item] = set()
            item_history[item].add(user)
            item_set.add(item)

    item_pair_num_no_common_rater = 0
    item_pair_num_with_common_rater = 0
    sp_no_common_rater = dict()
    sp_with_common_rater = dict()

    while True:
        item1, item2 = np.random.choice(list(item_set), size=2, replace=False)
        if item_pair_num_no_common_rater == NUM and item_pair_num_with_common_rater == NUM:
            break
        if item_pair_num_no_common_rater < NUM and len(item_history[item1] & item_history[item2]) == 0:
            item_pair_num_no_common_rater += 1
            if not nx.has_path(kg, item1, item2):
                sp = 'infinity'
            else:
                sp = nx.shortest_path_length(kg, item1, item2)
            if sp not in sp_no_common_rater:
                sp_no_common_rater[sp] = 0
            sp_no_common_rater[sp] += 1
            print(item_pair_num_no_common_rater, item_pair_num_with_common_rater)
        if item_pair_num_with_common_rater < NUM and len(item_history[item1] & item_history[item2]) > 0:
            item_pair_num_with_common_rater += 1
            if not nx.has_path(kg, item1, item2):
                sp = 'infinity'
            else:
                sp = nx.shortest_path_length(kg, item1, item2)
            if sp not in sp_with_common_rater:
                sp_with_common_rater[sp] = 0
            sp_with_common_rater[sp] += 1
            print(item_pair_num_no_common_rater, item_pair_num_with_common_rater)

    print(sp_no_common_rater)
    print(sp_with_common_rater)
