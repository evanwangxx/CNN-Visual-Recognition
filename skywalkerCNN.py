import tensorflow as tf
import numpy as np
import dataLoad
import os

picture = dataLoad.picture_scale
label = dataLoad.labels_onehot

picture_test = dataLoad.picture_test_scale
label_test = dataLoad.labels_test_onehot

img_shape = dataLoad.picture.shape
keep_prob = 0.6
epochs=5
batch_size=64

INCEPTION_LOG_DIR = './tmp/inception_v3_log2'
if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)

inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name = "inputs")
targets = tf.placeholder(tf.float32, [None, 10], name = "targets")


# network
info = ''' 
              input:   32 * 32 * 3 
       first filter:   32 * 32 * 64     2 * 2 | padding: same | relu        |
            pooling:   16 * 16 * 64     2 * 2 | padding: same | max-pooling | 
      second filter:   16 * 16 * 128    4 * 4 | padding: same | relu        | 
            pooling:    8 * 8  * 64     2 * 2 | padding: same | max-pooling |
full connection - 1:    1 * 1024                              | relu        |
full connection - 2:    1 * 512                               | relu        |
full connection - 3:    1 * 10                                | relu        |
             logits:    1 * 10
'''

layer_1_filter = tf.layers.conv2d(inputs = inputs,
                                  filters = 64,
                                  kernel_size = (2, 2),
                                  strides = (1, 1),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

layer_2_pooling = tf.layers.max_pooling2d(inputs = layer_1_filter,
                                          pool_size = (2, 2),
                                          strides = (2, 2),
                                          padding = 'same')


layer_3_filter = tf.layers.conv2d(inputs = layer_2_pooling,
                                  filters = 128,
                                  kernel_size = (2, 2),
                                  strides = (2, 2),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

layer_4_pooling = tf.layers.max_pooling2d(inputs = layer_3_filter,
                                          pool_size = (2, 2),
                                          strides = (1, 1),
                                          padding = 'same')

layer_5_filter = tf.layers.conv2d(inputs = layer_4_pooling,
                                  filters = 128,
                                  kernel_size = (4, 4),
                                  strides = (2, 2),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

layer_6_pooling = tf.layers.max_pooling2d(inputs = layer_5_filter,
                                          pool_size = (2, 2),
                                          strides = (1, 1),
                                          padding = 'same')

shape_layer4 = np.prod(layer_6_pooling.get_shape().as_list()[1:])
# print(shape_layer4)
layer_5_reshape = tf.reshape(tensor= layer_6_pooling,
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
                                                activation_fn= None)

layer_9_logit = tf.identity(layer_8_fc3, name='logits_')

#cost and  accuracy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_9_logit, labels=targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(layer_9_logit, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

cost_summary = tf.summary.scalar('cost', cost)
acc_summary = tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge([cost_summary, acc_summary])

count = 0
with tf.Session() as sess:
    print(info)
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(INCEPTION_LOG_DIR, sess.graph)

    for epoch in range(epochs):

        print("Epoch: ", epoch)

        for batch_i in range(img_shape[0] // batch_size - 1):

            feature_batch = picture[batch_i * batch_size: (batch_i + 1) * batch_size]
            label_batch = label[batch_i * batch_size: (batch_i + 1) * batch_size]

            train_loss, _, sum, val_acc = sess.run([cost, optimizer, summary_op, accuracy],
                                     feed_dict={inputs: feature_batch,
                                                targets: label_batch})

            test_acc = sess.run(accuracy,
                               feed_dict={inputs: picture_test,
                                          targets: label_test})
            if (count % 10 == 0):
                print(str(count) + ' | Train Loss {:.8f} | Val Acc {:4f} | Test Acc {:4f}'
                      .format(train_loss, val_acc, test_acc))

            count += 1

    summary_writer.close()

    saver = tf.train.Saver().save(sess, "./test_cifar2")