import tensorflow as tf
from aggregators import SumAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGNN_LS(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset):
        self._parse_args(args, adj_entity, adj_relation, interaction_table, offset) # interaction_table defined in train.py with key: user_id * 10^offset + item_id and value: y_{user_id, item_id} if the user-item pair is not found then the label is set to 0.5
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    # initialize weight matrices with xavier method to avoid vanishing or exploding gradients during training
    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        # LS regularization
        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr

    # constructs placeholders for tensors of user indices, item indices and labels
    # shape is set to [None], indicating that the size can vary from 1D
    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

#sets up the key components of the KGNN model: user, entity, and relation embeddings, aggregation of neighbor information, label smoothness regularization, computation of scores for user-item interactions
    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGNN_LS.get_initializer(), name='user_emb_matrix') # the embedding matrix for users with shape [n_user, dim]
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGNN_LS.get_initializer(), name='entity_emb_matrix') # the embedding matrix for entities with shape [n_entity, dim]
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGNN_LS.get_initializer(), name='relation_emb_matrix') # the embedding matrix for relation with shape [n_relation, dim]

        # [batch_size, dim]
        # creates a lookup to retrieve the embeddings for users based on their indices
        # and produces a tensor of shape [batch_size, dim] containing the embeddings for the users in the current batch
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices) # lists of the 1st, 2nd etc. order of neighbour entities of given items

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # LS regularization
        self._build_label_smoothness_loss(entities, relations)  # self.predicted_labels = ...

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

#takes a tensor of item indices (seeds) and iteratively retrieves neighbor entities and relations from the adjacency matrices (adj_entity, adj_relation).
    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1) # expands the dimensions of the seeds (=item_indices) tensor by adding a new axis at position 1
        entities = [seeds] #  is initialized with the seeds, representing the initial set of items
        relations = []
        for i in range(self.n_iter):
            # tf.gather is used to extract neighbor indices from adj_entity and adj_relation based on the current set of entities
            # tf.reshape is then applied to flatten the gathered neighbor entities and relations, resulting in 2D tensors with shapes [batch_size, -1].
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            # Appends the flattened neighbor entities and relations to their respective lists
            # This prepares the information for the next iteration, where the neighbor entities of the current iteration become the seeds for the next
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        # Returns the final lists entities and relations, containing neighbor information for each iteration.
        # The resulting structure is a list of tensors, where entities[i] and relations[i] correspond to the neighbor information at iteration i
        return entities, relations

    # feature propagation, performs aggregation of entity vectors based on neighbor vectors and relations
    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        #Constructs lists entity_vectors and relation_vectors containing the corresponding embeddings for each entity and relation in the input.
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            #If it's the last iteration (i == self.n_iter - 1), the aggregator is instantiated with a tanh activation function otherwise, it's instantiated with the default (no activation function).
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)  # TODO: either `lambda x:x` or `nn.tanh` (default)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = [] #store the aggregated entity vectors for the next iteration.
            for hop in range(self.n_iter - i): #Iterates over the hops (from 0 to self.n_iter - i - 1)
                shape = [self.batch_size, -1, self.n_neighbor, self.dim] #Constructs the shape for reshaping the neighbor vectors.
                #The -1 allows TensorFlow to determine the size of that dimension dynamically based on the sizes of the other dimensions and the total number of elements in the input tensor
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None) #aggregate entity vectors based on neighbor vectors and relations
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter #Updates entity_vectors with the aggregated entity vectors for the next iteration.

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim]) #Reshapes the aggregated entity vectors from the first hop into a 2D tensor with shape [self.batch_size, self.dim].

        return res, aggregators

    # LS regularization
    def _build_label_smoothness_loss(self, entities, relations):

        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            users = tf.expand_dims(self.user_indices, 1)
            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter #Calculate the concatenated user-entity indices.

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(user_entity_concat) #Look up the initial label for each user-entity pair from the interaction table.
            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out, Create a holdout mask to identify held-out items, indicating whether an item is held out or not (not included in the training data)
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item, Create a reset mask to identify labeled entities, indicating whether the entity is a labeled item (initial label is not 0.5)
            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # Initializes the label based on the holdout mask. If an item is held out, its label is set to the initial label; otherwise, it is set to 0.5

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

        # label propagation
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations] #Retrieves relation vectors based on the given relations
        aggregator = LabelAggregator(self.batch_size, self.dim) #Initialize a label aggregator
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                #Aggregate neighbor labels using the aggregator function
                vector = aggregator(self_vectors=entity_labels[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_labels[hop + 1], [self.batch_size, -1, self.n_neighbor]),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], [self.batch_size, -1, self.n_neighbor, self.dim]),
                                    user_embeddings=self.user_embeddings,
                                    masks=reset_masks[hop])
                entity_labels_next_iter.append(vector)
            entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1) #Extract the final predicted labels from the last iteration's entity labels

# It calculates the loss function, consisting of three components: base loss, L2 loss, and LS (label smoothness) loss, and sets up an optimizer to minimize this combined loss.
    def _build_train(self):
        # base loss
        #Calculates the base loss using sigmoid cross-entropy between the labels and scores (logits) predicted by the model
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # L2 loss
        #Calculates the L2 regularization loss for the user, entity, and relation embedding matrices.
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            #it computes the L2 loss for the weights of all aggregators
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        #Combines the base loss and the L2 regularization loss by weighting the latter with a hyperparameter l2_weight.
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # LS loss
        #Calculates the LS regularization loss using sigmoid cross-entropy between the labels and the predicted labels from label propagation
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))
        #Adds the LS regularization loss to the combined loss, weighted by the hyperparameter ls_weight.
        self.loss += self.ls_weight * self.ls_loss  # TODO: comment or uncomment

        #Sets up an Adam optimizer to minimize the combined loss (loss) during training with a given learning rate (lr).
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        #returns this list as its output, providing the optimizer result and the corresponding loss value obtained during training
        return sess.run([self.optimizer, self.loss], feed_dict) #feed_dict=dictionary mapping TensorFlow placeholders to their values

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict) #retrieves the true labels (labels) and the model's predicted scores (scores_normalized) for the given input data.
        auc = roc_auc_score(y_true=labels, y_score=scores) #measures the ability of the model to distinguish between positive and negative instances
        #converts the continuous predicted scores into binary predictions
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
