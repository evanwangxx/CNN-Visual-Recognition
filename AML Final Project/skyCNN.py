import tensorflow as tf
import numpy as np
import dataLoad
import os

x_train = dataLoad.picture_scale
label = dataLoad.labels_onehot

picture_test = dataLoad.picture_test_scale
label_test = dataLoad.labels_test_onehot

img_shape = x_train.shape
keep_prob = 0.6
epochs=5
batch_size=64

inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name = "inputs")
targets = tf.placeholder(tf.float32, [None, 10], name = "targets")

layer_1_filter = tf.layers.conv2d(inputs = inputs,
                                  filters = 64,
                                  kernel_size = (2, 2),
                                  strides = (2, 2),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

layer_2_pooling = tf.layers.max_pooling2d(inputs = layer_1_filter,
                                          pool_size = (2, 2),
                                          strides = (2, 2),
                                          padding = 'same')


layer_3_filter = tf.layers.conv2d(inputs = layer_2_pooling,
                                  filters = 128,
                                  kernel_size = (4, 4),
                                  strides = (2, 2),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

layer_4_pooling = tf.layers.max_pooling2d(inputs = layer_3_filter,
                                          pool_size = (2, 2),
                                          strides = (2, 2),
                                          padding = 'same')


shape_layer4 = np.prod(layer_4_pooling.get_shape().as_list()[1:])
layer_5_reshape = tf.reshape(tensor= layer_4_pooling,
                             shape = [-1, shape_layer4])

layer_6_fc = tf.contrib.layers.fully_connected(inputs = layer_5_reshape,
                                               num_outputs = 1024,
                                               activation_fn=tf.nn.relu)
layer_6_fc = tf.nn.dropout(layer_6_fc, keep_prob)  # Faster with drop out

layer_7_fc2 = tf.contrib.layers.fully_connected(inputs = layer_6_fc,
                                               num_outputs = 512,
                                                activation_fn=tf.nn.relu)

layer_8_fc3 = tf.contrib.layers.fully_connected(inputs = layer_7_fc2,
                                                num_outputs = 10,
                                                activation_fn=None)

layer_9_logit = tf.identity(layer_8_fc3, name='logits_')



# cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_9_logit, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
#
# accuracy
correct_pred = tf.equal(tf.argmax(layer_9_logit, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for batch_i in range(img_shape[0] // batch_size - 1):

        feature_batch = x_train[batch_i * batch_size: (batch_i + 1) * batch_size]
        label_batch = label[batch_i * batch_size: (batch_i + 1) * batch_size]

        train_loss, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs: feature_batch,
                                            targets: label_batch})

        val_acc = sess.run(accuracy,
                           feed_dict={inputs: picture_test,
                                      targets: label_test})

        if (count % 10 == 0):
            print(str(count) + ' | Train Loss {:.8f}  | Accuracy {:4f}'.format(train_loss, val_acc))

        count += 1
