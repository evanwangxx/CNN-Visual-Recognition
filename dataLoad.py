''' Load Data'''

import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

# from sklearn.model_selection import train_test_split

# unpickle date from the path
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict

# load training data
def load_train_data(path):
    feature = np.empty(shape=(10000, 3072))
    label = list()
    for i in range(1, 6):
        path_tmp = path + "/data_batch_" + str(i)
        print("-- Info: load data - " + path_tmp)

        data_tmp = unpickle(path_tmp)

        feature_tmp = data_tmp[b'data']
        label_tmp = data_tmp[b'labels']

        feature = np.concatenate((feature, feature_tmp), axis=0)
        label = label + label_tmp

    feature = feature[10000:60000, :]

    return(feature, label)

# load testing data
def load_test_data(path):

    data = unpickle(path + "/test_batch")

    feature = data[b'data']
    label= data[b'labels']

    return(feature, label)

# reshape features into picture 3 * 32 * 32
def data_reshape(feature):

    pic = feature.reshape((len(feature)), 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

    return pic

# Split data into training & test set
def data_split(ratio = 0.8):
    return None

def onehot(label, n):

    onehot = LabelBinarizer().fit(np.array(range(n)))

    onehot_label = onehot.transform(label)

    return onehot_label

def scale(data):

    data_reshape = data.reshape(data.shape[0], 32 * 32 * 3)

    data_scale = MinMaxScaler().fit_transform(data_reshape)

    data_scale_reshape = data_scale.reshape(data_scale.shape[0], 32, 32, 3)

    return data_scale_reshape


data_path = "/Users/hongbowang/Personal/Data/cifar-10-batches-py"

feature_train, label_train = load_train_data(data_path)
feature_test, label_test = load_test_data(data_path)

picture = data_reshape(feature_train)
picture_scale = scale(picture)
labels_onehot = onehot(label_train, 10)

picture_test = data_reshape(feature_test)
picture_test_scale = scale(picture_test)
labels_test_onehot = onehot(label_test, 10)

print(picture.shape, labels_onehot.shape)
print(picture_test.shape, labels_test_onehot.shape)