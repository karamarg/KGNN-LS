import tensorflow as tf
import numpy as np
from model import KGNN_LS


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    # print(f"n_user={n_user}, n_item={n_item}, n_entity={n_entity}, n_relation={n_relation}") #n_user=1872, n_item=3846, n_entity=9366, n_relation=60 for books
    # adj_entity & adj_relation: num of rows = num of entities and num of cols = num of neighbors, if neighbors more than given num then choose randomly, if less then select with replacement
    # print(adj_entity) # [[4454 4454 4454 ... 4454 4454 4454]......[2241 2241 2241 ... 2241 2241 2241]] rows depicting the entities and columns the connected entities
    # print(adj_relation) # [[0 0 0 ... 0 0 0]....[8 7 8 ... 8 8 8]] rows depicting the entities and columns the relation type
    # print(train_data) #[userID | item_index | target(0/1)] i.e. [[0   20    1][0   22    1]....[1871 1410    0]]
    interaction_table, offset = get_interaction_table(train_data, n_entity)
    # print(interaction_table) #<tensorflow.contrib.lookup.lookup_ops.HashTable object at 0x000001D8E9E01F60> contains keys with combined user+item ids and values 0/1 of the target see get_interaction_table() below
    model = KGNN_LS(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)
    # print(model) #<model.KGNN_LS object at 0x000002720E80B588>

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    # print(user_list) # [1140  750 1797...] users selected from the intersection of train and test users
    # print(train_record) # {0: {2432, 2398, 259, 963, 3724, 2995, 20, 3669, 22, 23, 24, 3796, 26, 27, 28, 30, 31}, 1: {33, 1115},...} all positive and negative items of each user in the train set
    # print(test_record) # {430: {105, 1757, 1758}, 1079: {1256, 522, 994},...} all positive items of each user in the test set
    # print(item_set) # {0,1,2,...,3844, 3845}
    # print(k_list) # [1, 3, 10] the distinct top k

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.init.run()

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size)) # the corresponding loss value obtained during training for the minibatch provided as dictionary of concatenated user_indices+item_indices as keys and labels as values from get_feed_dict() below
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            """test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
            train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
            eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
            """

            if step == args.n_epochs-1:
                test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
                precision, recall = topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print(f'{test_auc:.3f} \t {test_f1:.3f} \t {precision[0]:.3f} \t {precision[1]:.3f} \t {precision[2]:.3f} \t {recall[0]:.3f} \t {recall[1]:.3f} \t {recall[2]:.3f}')

            """
            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                  % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')
            """


# interaction_table is used for fetching user-item interaction label in LS regularization
# key: user_id * 10^offset + item_id
# value: y_{user_id, item_id}
def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    # print(keys) #[      20       22       23 ... 18711630 18711394 18711410] keys are created by combining the user and the item ids
    # print(len(keys)) # 25408 for books
    values = train_data[:, 2].astype(np.float32)
    # print(values) #[1. 1. 1. ... 0. 0. 0.]
    # print(set(values)) #{0.0, 1.0}

    interaction_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5) # if a key is not present in the hash table during a lookup, the hash table will return the default value of 0.5
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item): # train_data or train_data [userID | item_index | target(0/1)] i.e. [[0   20    1][0   22    1]....[1871 1410    0]]
    if show_topk:
        user_num = 100 #the maximum number of users to consider for evaluation to 100.
        #k_list = [1, 2, 5, 10, 20, 50, 100]  # TODO: changed this, but keep the following
        k_list = [1, 3, 10]
        train_record = get_user_record(train_data, True) #Generates a dictionary of user interaction records for training data, considering both negative and  positive labels (True).
        # print(train_data)
        # print(train_data[train_data[:, 0] == 0]) # it returns all the user|item|label triplets for user 0, used to check if the train record contains both positive and negative items for each user
        # print(train_record) # {0: {2432, 2398, 259, 963, 3724, 2995, 20, 3669, 22, 23, 24, 3796, 26, 27, 28, 30, 31}, 1: {33, 1115}
        test_record = get_user_record(test_data, False) #Generates a dictionary of user interaction records for testing data, considering all labels (False)
        # print(test_data[test_data[:, 0] == 305]) # for users in the test data only the positive items are kept
        # print(test_record) # {430: {105, 1757, 1758},305: {184}...
        user_list = list(set(train_record.keys()) & set(test_record.keys())) #intersection of user IDs between training and testing data.
        # np.random.set_seed(0) TODO: add seed for np.random if hypothesis tests end up being not confident
        # print(f"train_users={len(set(train_record.keys()))},test_users={len(set(test_record.keys()))},common_users={len(user_list)}") # train_users=1870,test_users=1641,common_users=1640
        # the following prints show that a user can be both in the train and in the test set but for different item interactions. in the train sets all positive and negative items are included but in the test set only the positive items
        # print(train_data[train_data[:, 0] == 0])
        # print(train_record[0])
        # print(test_data[test_data[:, 0] == 0])
        # print(test_record[0])
        if len(user_list) > user_num: #If the number of users exceeds user_num, randomly selects user_num users from the intersection without replacement.
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item))) # Creates a set containing all item IDs in the dataset.
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5 #Returns a list of None values five times, representing an empty configuration.


#creates a dictionary called feed_dict where keys are placeholders in the TensorFlow model
#(model.user_indices, model.item_indices, model.labels) and values are slices of data corresponding to the specified range [start:end, ...]
def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    # print(feed_dict) # {<tf.Tensor 'user_indices:0' shape=(?,) dtype=int64>: array([1228,..,317], dtype=int64), <tf.Tensor 'item_indices:0' shape=(?,) dtype=int64>: array([2954,...765], dtype=int64), <tf.Tensor 'labels:0' shape=(?,) dtype=float32>: array([0,...,1], dtype=int64) }
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = [] # lists to store the calculated AUC (Area Under the Curve) and F1 scores for each batch.
    f1_list = []
    while start + batch_size <= data.shape[0]: #total rows of data
        auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)) #Calculates the mean AUC and F1 scores across all batches and returns them as floating-point numbers.


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    #Initialize empty dictionaries to store precision and recall values for different values of K
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user]) # list of items for the current user by excluding items the user interacted with in the training set.
        item_score_map = dict() #empty dictionary to store item-score pairs.
        start = 0
        # this while creates a dictionary with the items as keys and the corresponding scores as values for each user
        while start + batch_size <= len(test_item_list):
            #Use the recommendation model to get scores for the test items in the current batch.
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size, #the feed dictionary contains the current user's ID repeated batch_size times and the batch of test item indices
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            # print(user) # [955]
            # print(items) # [3601 3602 3603 3604
            # print(scores) # [0.7424231  0.7260689 .....]
            for item, score in zip(items, scores): # zip iterates in corresponding items in two lists
                item_score_map[item] = score #Store the item-score pairs in the item_score_map dict
            start += batch_size
            # print(item_score_map) # {0: 0.5671108, 1: 0.086631685, 2: 0.26225865,....,3727: 0.09711447, 3728: 0.10478622}

        # padding the last incomplete minibatch if exists, By handling the remaining test items in this way, the code ensures that all test items are processed and their scores are stored in the item_score_map dictionary for further evaluation
        #Use the recommendation model to get scores for the remaining test items and pad the batch with the last item score.
        if start < len(test_item_list): #Check if there are remaining test items after processing them in batches.
            items, scores = model.get_scores( #If the number of remaining test items is less than batch_size, pad the batch with duplicates of the last test item to match the batch size. This ensures consistency in the batch size for processing.
                sess, {model.user_indices: [user] * batch_size, #Pass a feed dictionary containing the current user's ID repeated batch_size times and the remaining test item indices.
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        #Sort the item-score pairs in descending order of scores and extract the sorted item list
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        # print(item_sorted) #[96, 36, 87, 46, 118, 55, 105, 52, 74,...]
        # print(len(item_sorted)) # 3829

        # this creates dictionaries with k as keys and lists that contain the calculated kpis for each user as values i.e. {1: [0.0, 0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0, 0.0], 10: [0.1, 0.0, 0.0, 0.0]} the precision for the first 4 users for k=1,3,10
        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user]) #Calculate the number of correctly recommended items (hits) within the top-K recommendations for the current user
            precision_list[k].append(hit_num / k) #Calculate precision and recall for the current value of K and append them to the respective lists
            recall_list[k].append(hit_num / len(test_record[user]))
            # print(user)
            # print(k)
            # print(precision_list)
            # print(len(precision_list[k]))

    #Calculate the mean precision and recall for all values of K i.e. precision at k=1 is the mean of all the users
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall

#  It returns a dictionary user_history_dict where the keys are user IDs and the values are sets of items
#  that each user has interacted with for users in the test set or all positive and negative items for users in the training set.
def get_user_record(data, is_train): # data can be train or test data [userID | item_index | target(0/1)]
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1: #checks if either the data is for training or if the label of the interaction is positive (1). In either case, the interaction is considered valid for inclusion in the user's history.
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
