from __future__ import print_function

import os
import glob
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import reuters
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

# I/P - LSTM Shape : [samples, steps]
def encoder(x, emb_len, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = Embedding(top_words, emb_len, input_length=steps)
        net = LSTM(128, name='lstm')(x)
        net = Dense(z_dim)(net)
        net = tf.contrib.layers.flatten(net)
        return net

def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


n_epochs = 20
n_iters = 100
n_totclass = 46
n_support = 5
n_query = 5
n_examples = 18
top_words, steps, emb_len = 5000, 150, 32
z_dim = 64

# Load Train Dataset
# To load the dataset, create a vocabulary of the words and provide them indices at ther position
# Shape : np.array(list_0, list_1, ...) where each list is a sentence.
# After padding it using keras preprocessing, we get np.array([no_examples, max_steps])
# Pass this to embedding layer in keras to get word vectors for each.


'''
# NLP data - will modify this for the req

root_dir = './data/dataset_name'
train_split_path = os.path.join(root_dir, 'splits', 'train.txt')

with open(train_split_path, 'r') as train_split:
    train_classes = [line.rstrip() for line in train_split.readlines()]

n_classes = len(train_classes)
train_dataset = np.zeros([n_classes, n_examples, steps], dtype=np.float32)

# ... then do the rest based on the dataset.

'''

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=top_words)

x_train = sequence.pad_sequences(x_train, maxlen=steps)

n_classes = n_totclass
train_dataset = np.zeros([n_classes, n_examples, steps], dtype=np.float32)

for i in range(n_classes):
    indices = np.where(y_train == i)[0]
    class_vec = x_train[indices]
    for j in range(min(n_examples, indices.shape[0])):
        train_dataset[i, j] = class_vec[j].astype(np.float32)

print(train_dataset.shape)


# X - Support data : used to calculate the prototypes (mean of embeddings)
# Q - Query data   : used to optimize the cross_entropy loss

# The data is arranged in the format : [class_it_belongs_to, samples, img_h, img_w, img_] ...
# ... for our simplicity. When passed to encoder, it's reshaped to [no_images, img_h, img_w, img_c] ...
# ... by multiplying the first 2 dimensions.


# For NLP : [class, samples, steps]
x = tf.placeholder(tf.float32, [None, None, steps])
q = tf.placeholder(tf.float32, [None, None, steps])

x_shape = tf.shape(x)
q_shape = tf.shape(q)
num_classes, num_support = x_shape[0], x_shape[1]
num_queries = q_shape[1]

y = tf.placeholder(tf.int64, [None, None])
y_one_hot = tf.one_hot(y, depth=num_classes)

emb_x = encoder(tf.reshape(x, [num_classes * num_support, steps]), emb_len, z_dim)
emb_dim = tf.shape(emb_x)[-1]
emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)

emb_q = encoder(tf.reshape(q, [num_classes * num_queries, steps]), emb_len, z_dim, reuse=True)
dists = euclidean_distance(emb_q, emb_x)
log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])

ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))

train_op = tf.train.AdamOptimizer().minimize(ce_loss)

sess = tf.InteractiveSession()
K.set_session(sess)
sess.run(tf.global_variables_initializer())

for ep in range(n_epochs):
    for epi in range(n_iters):
        
        epi_classes = np.random.permutation(n_classes)[:n_totclass]
        support = np.zeros([n_totclass, n_support, steps], dtype=np.float32)
        query = np.zeros([n_totclass, n_query, steps], dtype=np.float32)
        
        for i, epi_cls in enumerate(epi_classes):
            selected = np.random.permutation(n_examples)[:n_support + n_query]
            support[i] = train_dataset[epi_cls, selected[:n_support]]
            query[i] = train_dataset[epi_cls, selected[n_support:]]
            
        support = np.expand_dims(support, axis=-1)
        query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(n_totclass)[:, np.newaxis], (1, n_query)).astype(np.uint8)
            
        _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y:labels})
        if (epi+1) % 50 == 0:
            print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, epi+1, n_iters, ls, ac))

# TEST

'''
x_test = sequence.pad_sequences(x_test, maxlen=steps)


root_dir = './data/omniglot'
test_split_path = os.path.join(root_dir, 'splits', 'test.txt')

with open(test_split_path, 'r') as test_split:
    test_classes = [line.rstrip() for line in test_split.readlines()]

n_test_classes = len(test_classes)
test_dataset = np.zeros([n_test_classes, n_examples, steps, ], dtype=np.float32)

for i, tc in enumerate(test_classes):
    alphabet, character, rotation = tc.split('/')
    rotation = float(rotation[3:])
    im_dir = os.path.join(root_dir, 'data', alphabet, character)
    im_files = sorted(glob.glob(os.path.join(im_dir, '*.png')))
    
    for j, im_file in enumerate(im_files):
        im = 1. - np.array(Image.open(im_file).rotate(rotation).resize((, steps)), np.float32, copy=False)
        test_dataset[i, j] = im

print(test_dataset.shape)

n_test_iters = 1000
n_test_totclass = 20
n_test_support = 5
n_test_query = 15

print('Testing...')
avg_acc = 0.
for epi in range(n_test_iters):
    
    epi_classes = np.random.permutation(n_test_classes)[:n_test_totclass]
    support = np.zeros([n_test_totclass, n_test_support, steps, ], dtype=np.float32)
    query = np.zeros([n_test_totclass, n_test_query, steps, ], dtype=np.float32)
    
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(n_examples)[:n_test_support + n_test_query]
        support[i] = test_dataset[epi_cls, selected[:n_test_support]]
        query[i] = test_dataset[epi_cls, selected[n_test_support:]]
    
    support = np.expand_dims(support, axis=-1)
    query = np.expand_dims(query, axis=-1)
    labels = np.tile(np.arange(n_test_totclass)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
    
    ls, ac = sess.run([ce_loss, acc], feed_dict={x: support, q: query, y:labels})
    avg_acc += ac
    
    if (epi+1) % 50 == 0:
        print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_iters, ls, ac))

avg_acc /= n_test_iters
print('Average Test Accuracy: {:.5f}'.format(avg_acc))
'''