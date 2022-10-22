import numpy as np
import pandas as pd
import sys


class DMF_input(object):
    def __init__(self):
        self.data, self.train_features, self.test_features, self.shape = self.load_data()
        self.trainDict = self.data_dict()
        # self.testDict = self.data_dict(self.test_features)

    def load_data(self):
        # DATA_DIR1 = 'data/IV/trainset(IV).csv'
        # DATA_DIR2 = 'data/IV/testset(IV).csv'
        # DATA_DIR1 = 'data/DM/trainset(DM).csv'
        # DATA_DIR2 = 'data/DM/testset(DM).csv'
        DATA_DIR1 = 'data/AA/trainset(AA).csv'
        DATA_DIR2 = 'data/AA/testset(AA).csv'
        COLLUMN_NAME = ['user', 'item', 'label']
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

        train_data.user = train_data['user'] - 1
        train_data.item = train_data['item'] - 1
        test_data.user = test_data['user'] - 1
        test_data.item = test_data['item'] - 1

        train_features = train_data
        test_features = test_data
        data = pd.concat([train_features, test_features], axis=0, ignore_index=True)

        return data, train_features, test_features, [user_size, item_size]

    def data_dict(self):
        trainDict = {}
        for i in range(len(self.train_features)):
            u = self.train_features['user'][i]
            trainDict.setdefault(u, {})
            t = self.train_features['item'][i]
            r = self.train_features['label'][i]
            trainDict[u].setdefault(t, r)
        return trainDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]])
        for i in range(len(self.train_features)):
            u = self.train_features['user'][i]
            t = self.train_features['item'][i]
            r = self.train_features['label'][i]
            train_matrix[u, t] = r
        print('Embedding shape:', train_matrix.shape)
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rating = []
        for i in range(len(data)):
            user.append(data['user'][i])
            item.append(data['item'][i])
            rating.append(data['label'][i])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (data['user'][i], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                    user.append(data['user'][i])
                    item.append(j)
                    rating.append(0)

        return np.array(user), np.array(item), np.array(rating)

    def test_neg(self, test):
        user = []
        item = []
        rating = []
        user_tmp = []
        item_tmp = []
        rating_tmp = []
        for i in range(1, len(test) + 1):
            if i < len(test):
                u = test['user'][i - 1]
                t = test['item'][i - 1]
                r = test['label'][i - 1]
                v = test['user'][i]
                if u == v:
                    user_tmp.append(u)
                    item_tmp.append(t)
                    rating_tmp.append(r)
                else:
                    user_tmp.append(u)
                    item_tmp.append(t)
                    rating_tmp.append(r)
                    user.append(user_tmp)
                    item.append(item_tmp)
                    rating.append(rating_tmp)
                    user_tmp = []
                    item_tmp = []
                    rating_tmp = []
            else:
                u = test['user'][i - 1]
                t = test['item'][i - 1]
                r = test['label'][i - 1]
                user_tmp.append(u)
                item_tmp.append(t)
                rating_tmp.append(r)
                user.append(user_tmp)
                item.append(item_tmp)
                rating.append(rating_tmp)
        return [np.array(user), np.array(item), np.array(rating)]
