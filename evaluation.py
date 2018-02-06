import random
import tensorflow as tf
import dataLoad
import skywalkerCNN

picture_test = dataLoad.picture_test
label_test = dataLoad.labels_test_onehot

inputs = tf.placeholder(tf.float32, [None, 32, 32, 3], name = "input")
targets = tf.placeholder(tf.float32, [None, 10], name = "targets")

save_model_path = "./test_cifar"

loaded_graph = tf.Graph()
test_batch_size = 100
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(save_model_path + '.meta')
    loader.restore(sess, save_model_path)

    # 加载tensor
    loaded_x = loaded_graph.get_tensor_by_name('inputs:0')
    loaded_y = loaded_graph.get_tensor_by_name('targets:0')
    loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
    loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

    # 计算test的准确率
    test_batch_acc_total = 0
    test_batch_count = 0

    print("Begin test...")
    for batch_i in range(picture_test.shape[0] // test_batch_size - 1):
        test_feature_batch = picture_test[batch_i * test_batch_size: (batch_i + 1) * test_batch_size]
        test_label_batch = label_test[batch_i * test_batch_size: (batch_i + 1) * test_batch_size]
        test_batch_acc_total += sess.run(
            loaded_acc,
            feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch})
        test_batch_count += 1

    print('Test Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))



correct_pred = tf.equal(tf.argmax(layer_9_logit, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')