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

# Conv Pool
cp_1_1 = tf.layers.conv2d(inputs = inputs,
                                  filters = 96,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_1_2 = tf.layers.conv2d(inputs = cp_1_1,
                                  filters = 96,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_1_3 = tf.layers.conv2d(inputs = cp_1_2,
                                  filters = 96,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_mp1 = tf.layers.max_pooling2d(inputs = cp_1_3,
                                          pool_size = (3, 3),
                                          strides = (2, 2),
                                          padding = 'same')

cp_2_1 = tf.layers.conv2d(inputs = inputs,
                                  filters = 192,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_2_2 = tf.layers.conv2d(inputs = cp_1_1,
                                  filters = 192,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_2_3 = tf.layers.conv2d(inputs = cp_1_2,
                                  filters = 192,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

cp_mp2 = tf.layers.max_pooling2d(inputs = cp_2_3,
                                          pool_size = (3, 3),
                                          strides = (2, 2),
                                          padding = 'same')




# network
f1_conv = tf.layers.conv2d(inputs = cp_mp2,
                                  filters = 96,
                                  kernel_size = (5, 5),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f1_conv = tf.layers.conv2d(inputs = f1_conv,
                                  filters = 96,
                                  kernel_size = (1, 1),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f2_maxPool = tf.layers.max_pooling2d(inputs = f1_conv,
                                          pool_size = (3, 3),
                                          strides = (2, 2),
                                          padding = 'same')


f3_conv = tf.layers.conv2d(inputs = f2_maxPool,
                                  filters = 192,
                                  kernel_size = (5, 5),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f3_conv = tf.layers.conv2d(inputs = f3_conv,
                                  filters = 192,
                                  kernel_size = (1, 1),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f4_maxPool = tf.layers.max_pooling2d(inputs = f3_conv,
                                          pool_size = (3, 3),
                                          strides = (2, 2),
                                          padding = 'same')

f5_conv = tf.layers.conv2d(inputs = f4_maxPool,
                                  filters = 192,
                                  kernel_size = (3, 3),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f6_conv = tf.layers.conv2d(inputs = f5_conv,
                                  filters = 192,
                                  kernel_size = (1, 1),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

f7_conv = tf.layers.conv2d(inputs = f6_conv,
                                  filters = 10,
                                  kernel_size = (1, 1),
                                  padding = "same",
                                  activation= tf.nn.relu,
                                  kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

shape = np.prod(f7_conv.get_shape().as_list()[1:])
# print(shape_layer4)
f7_reshape = tf.reshape(tensor= f7_conv, shape = [-1, shape])
f8_fc = tf.contrib.layers.fully_connected(inputs = f7_reshape,
                                                num_outputs = 10,
                                                activation_fn= None)

#cost and  accuracy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f8_fc, labels=targets))
optimizer = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(f8_fc, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

cost_summary = tf.summary.scalar('cost', cost)
acc_summary = tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge([cost_summary, acc_summary])

count = 0
with tf.Session() as sess:
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

saver = tf.train.Saver().save(sess, "./test_cifarB")