import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file [userID | item_index | target(0/1)]
    rating_file = 'data/' + args.dataset + '/ratings_final' # '../data/' + args.dataset + '/ratings_final'
    # Check if a preprocessed numpy file exists, if not, load the text file and save it as a numpy file
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    # Compute the number of users and items
    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args) # Split the dataset into training, evaluation, and testing sets

    return n_user, n_item, train_data, eval_data, test_data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0] # number of total ratings

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False) # randomly select indices for the evaluation set
    left = set(range(n_ratings)) - set(eval_indices) # indices left after removing the ones that were chosen for evaluation
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False) # randomly select from the left indices for the test set
    train_indices = list(left - set(test_indices)) # from the left indices remove the test indices to create the train ones

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


# loads as np array the kg final dataset  [entity_id | relation_id | entity_id] and constructs the knowledge graph and the adjacency matrices of entities and relation
def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = 'data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2])) # number of total distinct entities in heads and tails
    n_relation = len(set(kg_np[:, 1])) # number of relations

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


# Define a function to construct the KG as a dictionary of entities and their relations
# i.e. {2086: [(3846, 0)], 3846: [(2086, 0), (1772, 0),…], ….}, {head/tail:[(tail/head,relationship), (tail/head,relationship),...]}
def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    # Iterate over each triple in the KG
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

# Define a function to construct adjacency matrices for entities and relations
def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        # print(neighbors) # [(6694, 0), (8716, 2), (8716, 1)] same neighbor but different relationship
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size: # if neighbors more than threshold, then sample indices of neighbors
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else: # if neighbors less than threshold, then sample indices of neighbors with replacement in order to have vectors of equal size
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices]) # replace the 0 value with the entity id
        # print(adj_entity[entity]) # [6694 8716 6694 8716 8716 8716 6694 8716]
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices]) # replace the 0 value with the relation id
        # print(adj_relation[entity]) # [0 2 0 1 2 1 0 2]

    return adj_entity, adj_relation

