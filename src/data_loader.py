import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    if args.sampling_method == 'random':
        adj_entity, adj_relation = construct_adj(args, kg, n_entity)
    elif args.sampling_method == 'degree':
        adj_entity, adj_relation = construct_adj_with_degree_weighted_sampling(args, kg, n_entity)
    elif args.sampling_method == 'relation':
        adj_entity, adj_relation = construct_adj_with_relation_frequency_weighted_sampling(args, kg, n_entity)
    elif args.sampling_method == 'degree_relation':
        adj_entity, adj_relation = construct_adj_with_degree_and_relation_frequency_weighted_sampling(args, kg, n_entity)
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation


def construct_adj_with_degree_weighted_sampling(args, kg, entity_num):
    print('constructing adjacency matrix with degree-weighted sampling ...')

    # Initialize adjacency matrices
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)

    for entity in range(entity_num):
        neighbors = kg.get(entity, [])
        if not neighbors:
            continue  # Skip entities with no neighbors

        # Calculate degrees for each neighbor
        neighbor_degrees = np.array([len(kg.get(neighbor[0], [])) for neighbor in neighbors], dtype=np.float32)

        # Normalize probabilities
        neighbor_probs = neighbor_degrees / np.sum(neighbor_degrees)

        # Sample neighbors based on degree-weighted probabilities
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=False,
                p=neighbor_probs
            )
        else:
            # Pad with replacement sampling if fewer than K neighbors
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=True,
                p=neighbor_probs
            )

        sampled_neighbors = [neighbors[i] for i in sampled_indices]
        adj_entity[entity] = [neighbor[0] for neighbor in sampled_neighbors]
        adj_relation[entity] = [neighbor[1] for neighbor in sampled_neighbors]

    return adj_entity, adj_relation


def construct_adj_with_relation_frequency_weighted_sampling(args, kg, entity_num):
    print('constructing adjacency matrix with relation-frequency sampling ...')

    # Count relation frequencies across the KG
    relation_counts = {}
    for neighbors in kg.values():
        for _, relation in neighbors:
            if relation not in relation_counts:
                relation_counts[relation] = 0
            relation_counts[relation] += 1

    # Normalize frequencies to create probabilities
    total_relations = sum(relation_counts.values())
    relation_probs = {rel: count / total_relations for rel, count in relation_counts.items()}

    # Initialize adjacency matrices
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)

    for entity in range(entity_num):
        neighbors = kg.get(entity, [])
        if not neighbors:
            continue  # Skip entities with no neighbors

        # Calculate probabilities based on relation frequency
        neighbor_probs = np.array([relation_probs[neighbor[1]] for neighbor in neighbors], dtype=np.float32)
        neighbor_probs /= np.sum(neighbor_probs)  # Normalize

        # Sample neighbors based on relation-frequency probabilities
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=False,
                p=neighbor_probs
            )
        else:
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=True,
                p=neighbor_probs
            )

        sampled_neighbors = [neighbors[i] for i in sampled_indices]
        adj_entity[entity] = [neighbor[0] for neighbor in sampled_neighbors]
        adj_relation[entity] = [neighbor[1] for neighbor in sampled_neighbors]

    return adj_entity, adj_relation


def construct_adj_with_degree_and_relation_frequency_weighted_sampling(args, kg, entity_num):
    print('constructing adjacency matrix with degree and relation-frequency sampling ...')

    # Count relation frequencies across the KG
    relation_counts = {}
    for neighbors in kg.values():
        for _, relation in neighbors:
            if relation not in relation_counts:
                relation_counts[relation] = 0
            relation_counts[relation] += 1

    # Normalize relation frequencies to create probabilities
    total_relations = sum(relation_counts.values())
    relation_probs = {rel: count / total_relations for rel, count in relation_counts.items()}

    # Initialize adjacency matrices
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)

    for entity in range(entity_num):
        neighbors = kg.get(entity, [])
        if not neighbors:
            continue  # Skip entities with no neighbors

        # Calculate probabilities based on degree and relation frequency
        neighbor_degrees = np.array([len(kg[neighbor[0]]) for neighbor in neighbors], dtype=np.float32)
        neighbor_relation_probs = np.array([relation_probs[neighbor[1]] for neighbor in neighbors], dtype=np.float32)

        # Combine the two probabilities with a weighted average
        alpha = 0.5  # Weight for degree vs relation frequency
        combined_probs = alpha * neighbor_degrees + (1 - alpha) * neighbor_relation_probs
        combined_probs /= np.sum(combined_probs)  # Normalize

        # Sample neighbors based on combined probabilities
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=False,
                p=combined_probs
            )
        else:
            sampled_indices = np.random.choice(
                range(n_neighbors),
                size=args.neighbor_sample_size,
                replace=True,
                p=combined_probs
            )

        sampled_neighbors = [neighbors[i] for i in sampled_indices]
        adj_entity[entity] = [neighbor[0] for neighbor in sampled_neighbors]
        adj_relation[entity] = [neighbor[1] for neighbor in sampled_neighbors]

    return adj_entity, adj_relation
