import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {} #dictionary which will store the layer IDs for different layers

#get the layer ID based on the layer name. If the layer name is not in the LAYER_IDS dictionary, it initializes it with a value of 0 and returns 0.
# If the layer name already exists, it increments the corresponding value in the dictionary and returns the incremented value.
def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]

# defines a class Aggregator and its subclasses SumAggregator and LabelAggregator for aggregating information from neighboring nodes
class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name): # act=activation function
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

#aggregates information from neighbor vectors based on the relations between the entities and user embeddings
    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim]) #batch_size represents the number of entities, and dim represents the dimensionality of the embeddings.

            # [batch_size, -1, n_neighbor]
            #computes the relevance scores of each neighbor relation with respect to the user embeddings.
            user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1) # element-wise multiplication between the reshaped user_embeddings tensor and the neighbor_relations tensor, followed by reducing the result along the last axis (axis=-1) by taking the mean
            user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1) #applies the softmax function along the last axis (dim=-1) of the user_relation_scores tensor, resulting in normalized scores representing the importance of each neighbor relation.

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            # aggregates the neighbor vectors based on their importance scores computed from the user embeddings and neighbor relations.
            neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2) #computes the element-wise multiplication between the normalized relation scores and the neighbor_vectors tensor, followed by reducing the result along the third axis (axis=2) by taking the mean.
        else:
            # [batch_size, -1, dim]
            #aggregating the neighbor vectors without considering the user embeddings or neighbor relations.
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2) #computes the mean of the neighbor_vectors tensor along the third axis

        return neighbors_aggregated


#Implements the aggregation operation by summing up the self vectors and neighbor vectors, applying dropout, and passing through a fully connected layer with a specified activation function.
class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None): # TODO: dropout=0.2 argument instead of 0.
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name) #This calls the constructor of the superclass Aggregator with the provided arguments.

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable( #initializes the weight matrix for the aggregator using the Xavier initializer
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias') #initializes the bias vector for the aggregator using zeros.

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        #aggregate information from neighbor vectors based on user embeddings and neighbor relations.
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim]) #computes the element-wise sum of the self vectors and the aggregated neighbor vectors. It then reshapes the resulting tensor to have shape [-1, self.dim], where -1 represents a dynamic size determined at runtime.
        # TODO: the above line was the default, maybe try with this
        # output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout) #applies dropout regularization to the output tensor to prevent overfitting. The keep_prob argument specifies the probability that each element is kept.
        output = tf.matmul(output, self.weights) + self.bias #applies a linear transformation to the output tensor using the weight matrix and bias vector

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim]) #reshapes the output tensor to its original shape

        return self.act(output)


#Implements the aggregation operation by calculating weighted averages of neighbor labels based on relation scores and user embeddings
class LabelAggregator(Aggregator):
    def __init__(self, batch_size, dim, name=None):
        #This calls the constructor of the superclass Aggregator with the provided arguments. It sets the dropout to 0 (no dropout) and does not specify an activation function.
        super(LabelAggregator, self).__init__(batch_size, dim, 0., None, name)

    #This method is responsible for performing the label aggregation operation. It takes as input the self labels (labels of the current entity), neighbor labels (labels of neighboring entities),
    #neighbor relations (embeddings of relations with neighboring entities), user embeddings (embeddings of the user), and masks (optional, used for masking certain entities during aggregation).
    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim]) #This reshapes the user embeddings tensor to have shape [batch_size, 1, 1, dim] in order to perform element-wise operations with neighbor relations

        # [batch_size, -1, n_neighbor]
        #computes the element-wise multiplication of the reshaped user embeddings tensor with the neighbor relations tensor. It then calculates the mean along the last axis (axis=-1) to obtain a score for each neighbor relation.
        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        #applies the softmax function along the last axis of the user_relation_scores tensor to normalize the scores, ensuring they sum up to 1.
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1]
        #computes the element-wise multiplication of the normalized relation scores with the neighbor labels tensor. It then calculates the mean along the last axis to aggregate the labels of neighboring entities.
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_labels, axis=-1)
        #combines the self labels and the aggregated neighbor labels using the provided masks. It uses the masks to selectively choose between the self labels and the aggregated neighbor labels for each entity.
        output = tf.cast(masks, tf.float32) * self_labels + tf.cast(
            tf.logical_not(masks), tf.float32) * neighbors_aggregated

        return output
