import numpy as np
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()
import os

# DATA_DIR1 = 'data/IV/trainset(IV).csv'
# DATA_DIR2 = 'data/IV/testset(IV).csv'
# DATA_PATH = 'data/IV/'
# DATA_DIR1 = 'data/DM/trainset(DM).csv'
# DATA_DIR2 = 'data/DM/testset(DM).csv'
# DATA_PATH = 'data/DM/'
DATA_DIR1 = 'data/AA/trainset(AA).csv'
DATA_DIR2 = 'data/AA/testset(AA).csv'
DATA_PATH = 'data/AA/'
# DATA_DIR1 = 'data/MT/trainset(MT).txt'
# DATA_DIR2 = 'data/MT/testset(MT).txt'
# DATA_PATH = 'data/MT/'
is_MT = False
COLLUMN_NAME = ['user', 'item', 'label']


# 重新映射索引
def re_index(s):
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map


# 加载数据
def load_data():
    if is_MT:
        train_data = pd.read_csv(DATA_DIR1, sep='\t', header=None, names=COLLUMN_NAME, \
                                 usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float32})
        test_data = pd.read_csv(DATA_DIR2, sep='\t', header=None, names=COLLUMN_NAME, \
                                usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float32})
    else:
        train_data = pd.read_csv(DATA_DIR1, sep=',', header=None, names=COLLUMN_NAME, \
                                 usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float32})
        test_data = pd.read_csv(DATA_DIR2, sep=',', header=None, names=COLLUMN_NAME, \
                                usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.float32})
    # 合并
    full_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    # full_data.user = full_data['user'] - 1
    user_set = set(full_data['user'].unique())
    item_set = set(full_data['item'].unique())
    #
    #
    user_size = len(user_set)
    item_size = len(item_set)
    #
    # item_map = re_index(item_set)
    #
    # full_data['item'] = full_data['item'].map(lambda x: item_map[x]) #item映射
    #
    # # item_set = set(full_data.item.unique())
    #
    # user_bought = {}
    # for i in range(len(full_data)):
    #     u = full_data['user'][i]
    #     t = full_data['item'][i]
    #     if u not in user_bought:
    #         user_bought[u] = []
    #     user_bought[u].append(t)

    # user_negative = {}
    # for key in user_bought:
    #     user_negative[key] = list(item_set - set(user_bought[key]))

    # user_length = full_data.groupby('user').size().tolist() #统计每个用户已评项目数

    # split_train_test = []
    #
    # #抽取每个用户最后一个已评项目为测试集，其余为训练集
    # for i in range(len(user_set)):
    #     for _ in range(user_length[i] - 1):
    #         split_train_test.append('train')
    #     split_train_test('test')
    # full_data['split'] = split_train_test
    #
    # train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)
    # test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
    #
    # del train_data['split']
    # del test_data['split']

    # labels = np.ones(len(train_data), dtype=np.int32)

    train_data.user = train_data['user'] - 1
    train_data.item = train_data['item'] - 1

    train_labels = []
    for i in range(len(train_data)):
        r = (train_data['label'][i] - 1) / (5 - 1)
        train_labels.append(r)
    del train_data['label']
    train_features = train_data

    test_data.user = test_data['user'] - 1
    test_data.item = test_data['item'] - 1
    test_labels = test_data['label'].tolist()
    del test_data['label']
    test_features = test_data

    return ((train_features, train_labels),
            (test_features, test_labels),
            (user_size, item_size))


# def add_negative(features, user_negative, labels, numbers, is_training):
#     feature_user, feature_item, labels_add, feature_dict = [], [], [], {}
#
#     for i in range(len(features)):
#         user = features['user'][i]
#         item = features['item'][i]
#         label = labels[i]
#
#         feature_user.append(user)
#         feature_item.append(item)
#         labels_add.append(label)
#
#         neg_samples = np.random.choice(user_negative[user], size=numbers, replace=False).tolist()
#
#         if is_training:
#             for k in neg_samples:
#                 feature_user.append(user)
#                 feature_item.append(k)
#                 labels_add.append(0)
#         else:
#             for k in neg_samples:
#                 feature_user.append(user)
#                 feature_item.append(k)
#                 labels_add.append(k)
#
#     feature_dict['user'] = feature_user
#     feature_dict['item'] = feature_item
#
#     return feature_dict, labels_add

# def dump_data(features, labels, user_negative, num_neg, is_training):
#     if not os.path.exists(DATA_PATH):
#         os.makedirs(DATA_PATH) #递归创建目录
#
#     features, labels = add_negative(features, user_negative, labels, num_neg, is_training)
#
#     data_dict = dict([('user', features['user']),
#                       ('item', features['item']),
#                       ('label', labels)])
#     #print(data_dict)
#     if is_training:
#         np.save(os.path.join(DATA_PATH, 'train_data.npy'), data_dict)
#     else:
#         np.save(os.path.join(DATA_PATH, 'test_data.npy'), data_dict)

def dump_data(features, labels, is_training):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)  # 递归创建目录

    data_dict = dict([('user', features['user']),
                      ('item', features['item']),
                      ('label', labels)])
    # print(data_dict)
    if is_training:
        np.save(os.path.join(DATA_PATH, 'train_data.npy'), data_dict)
    else:
        np.save(os.path.join(DATA_PATH, 'test_data.npy'), data_dict)


# def train_input_fn(features, labels, batch_size, user_negative, num_neg):
#     data_path = os.path.join(DATA_PATH, 'train_data.npy')
#     if not os.path.exists(data_path):
#         dump_data(features, labels, user_negative, num_neg, True)
#
#     data = np.load(data_path, allow_pickle=True).item()
#
#     dataset = tf.data.Dataset.from_tensor_slices(data)
#     dataset = dataset.shuffle(100000).batch(batch_size)
#
#     return dataset

def train_input_fn(features, labels, batch_size):
    data_path = os.path.join(DATA_PATH, 'train_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, True)

    data = np.load(data_path, allow_pickle=True).item()

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(100000).batch(batch_size)

    return dataset


def eval_input_fn(features, labels):
    data_path = os.path.join(DATA_PATH, 'test_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, False)

    data = np.load(data_path, allow_pickle=True).item()
    print("Loading testing data finished!")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(100)

    return dataset
