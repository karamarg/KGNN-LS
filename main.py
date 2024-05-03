import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train

np.random.seed(555)

parser = argparse.ArgumentParser() 

'''
# movie
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=16, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=1.0, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
'''

'''
# book
parser.add_argument('--dataset', type=str, default='book', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-5, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
'''


# TODO: change default values (e.g., on ls_weight and n_iter=layers)
# music
parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.1, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')


'''
# restaurant
parser.add_argument('--dataset', type=str, default='restaurant', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of l2 regularization')
parser.add_argument('--ls_weight', type=float, default=0.5, help='weight of LS regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
'''

show_loss = False
show_time = False
show_topk = True

t = time()

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

args = parser.parse_args()
data = load_data(args)
print('auc \t f1 \t pr@1 \t pr@3 \t pr@10 \t rec@1 \t rec@3 \t rec@10')
train(args, data, show_loss, show_topk)

if show_time:
    print('time used: %d s' % (time() - t))
