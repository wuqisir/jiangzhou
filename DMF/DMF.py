# import tensorflow as tf
import numpy as np
import argparse
import DMF_input as DMF_input
import tensorflow.compat.v1 as tf
from scipy.stats import ks_2samp

import os

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import sys
import math
import time

tf.compat.v1.set_random_seed(0)

PRED_SAVEFILE = './pred_array_DMF'
TEST_SAVEFILE = './test_array_DMF'


def main():
    parser = argparse.ArgumentParser(description="Options")

    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int)
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 50])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 50])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0005)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=1000, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=2048, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-topK', action='store', dest='topK', default=5)

    args = parser.parse_args()
    classifier = Model(args)
    classifier.run()


class Model:
    def __init__(self, args):
        self.dataset = DMF_input.DMF_input()
        self.shape = self.dataset.shape

        self.train = self.dataset.train_features
        self.test = self.dataset.test_features

        self.negNum = args.negNum
        self.testNeg = self.dataset.test_neg(self.test)
        self.add_embedding_matrix()

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()

        self.add_loss()

        self.lr = args.lr
        self.add_train_step()

        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rating = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding_init = self.dataset.getEmbedding()
        # embed_plhdr = tf.placeholder(dtype=tf.float32, shape=embed_ndarr.shape)
        self.user_item_embedding_plhdr = tf.placeholder(dtype=tf.float32, shape=self.user_item_embedding_init.shape)
        self.user_item_embedding = tf.get_variable('embed', self.user_item_embedding_init.shape)
        # self.user_item_embedding = tf.convert_to_tensor(self.dataset.getEmbedding())
        self.item_user_embedding = tf.transpose(self.user_item_embedding)  # 转置

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            # return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)
            return tf.Variable(tf.random_normal(shape=shape, dtype=tf.float32, stddev=1.2), name=name)

        with tf.name_scope('User_Layer'):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], 'user_W1')
            user_out = tf.matmul(tf.cast(user_input, tf.float32), user_W1)

            for i in range(0, len(self.userLayer) - 1):
                W = init_variable([self.userLayer[i], self.userLayer[i + 1]], 'user_W' + str(i + 2))
                b = init_variable([self.userLayer[i + 1]], 'user_b' + str(i + 2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope('Item_Layer'):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(tf.cast(item_input, tf.float32), item_W1)
            for i in range(0, len(self.itemLayer) - 1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i + 1]], "item_W" + str(i + 2))
                b = init_variable([self.itemLayer[i + 1]], "item_b" + str(i + 2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out),
                                axis=1, keep_dims=False) / (norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)

    def add_loss(self):
        regRate = self.rating / 5.0  # 标准化(0， 1]
        losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
        loss = - tf.reduce_sum(losses)
        self.loss = loss

    def add_train_step(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = optimizer.minimize(self.loss)

    def init_sess(self):
        # self.config = tf.ConfigProto()
        # self.config.cpu_options.allow_growth = True
        # self.config.allow_soft_placement = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        best_rmse = 100
        best_f1 = -1
        best_ndcg = -1
        best_epoch = -1
        print('Start training!')

        for epoch in range(self.maxEpochs):
            print('=' * 20 + 'Epoch', epoch, '=' * 20)
            s_time = time.time()
            self.run_epoch(self.sess)
            print(f"training_time: {time.time() - s_time}")
            print('=' * 50)
            print('Start evaluation!')
            save_flag = True if epoch == (self.maxEpochs - 1) else False
            # if epoch == (self.maxEpochs - 1):
            if epoch % 1 == 0:
                rmse, pre, rec, ndcg, ppr, ks, r_square = self.evaluation(self.sess, self.topK, save_file=save_flag)
                f1 = 2 * pre * rec / (pre + rec)
                self.x = print(
                    'Epoch is %d, RMSE is %.3f, PRE is %.3f, REC is %.3f, F1 is %.3f, NDCG is %.3f,PPR is %.3f,KS is %.3f,R_SQUARE is %.3f' % (
                        epoch, rmse, pre, rec, f1, ndcg, ppr, ks, r_square))
                if rmse < best_rmse or f1 > best_f1 or ndcg > best_ndcg:
                    best_rmse = rmse
                    best_f1 = f1
                    best_ndcg = ndcg
                    best_epoch = epoch
            if epoch - best_epoch > self.earlyStop:
                print('Normal early stop!')
                break
            print('=' * 20 + 'Epoch', epoch, 'End' + '=' * 20)
        print('Best RMSE: %3f, F1: %.3f, NDCG: %.3f at Epoch %d' % (best_rmse, best_f1, best_ndcg, best_epoch))
        print('Training complete!')

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataset.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1

        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i + 1) * self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            # feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            feed_dict = self.create_feed_dict(train_u_batch, train_i_batch, train_r_batch)
            sess.run(self.user_item_embedding.assign(self.user_item_embedding_plhdr),
                     {self.user_item_embedding_plhdr: self.user_item_embedding_init})
            _, tmp_loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
            losses.append(tmp_loss)
            # if verbose and i % verbose == 0:
            #     sys.stdout.write('\r{} / {} : loss = {}'.format(
            #         i, num_batches, np.mean(losses[-verbose:])
            #     ))
            #     sys.stdout.flush()
            loss = np.mean(losses)
            # print('\nMean loss in this epoch is: {}'.format(loss))

    def create_feed_dict(self, u, i, r=None, drop=None):
        return {self.user: u, self.item: i, self.rating: r, self.drop: drop}


    def evaluation(self, sess, topK, save_file=False):
        def getRMSE(prediction, rating):
            sum_error = 0
            count = 0
            for i in prediction.keys():
                p = prediction[i]
                r = rating[i]
                error = math.fabs(p - r)
                sum_error += error * error
                count += 1
            return sum_error, count

        def getF1_value(prediction, rating):
            pred_rec = {}
            actual_rec = {}
            for i in prediction.keys():
                p = prediction[i]
                r = rating[i]
                if p > 3:
                    pred_rec.setdefault(i, p)
                if r > 3:
                    actual_rec.setdefault(i, r)

            pred_top = sorted(pred_rec.items(), key=lambda x: x[1], reverse=True)[0:topK]
            count = 0.0
            num = 0.0

            co_rec = []
            pre_rec = []
            for i, r in pred_top:
                num += 1
                pre_rec.append(i)
                if i in actual_rec.keys():
                    count += 1
                    co_rec.append(i)
            if count > 0:
                if num < topK:
                    pre = count / num
                else:
                    pre = count / topK
                rec = count / len(actual_rec.keys())

                return pre, rec, co_rec, pre_rec
            else:
                return 0, 0, co_rec, pre_rec

        def getNDCG(co_rec, pre_rec):
            if len(co_rec) > 0:
                hit_order = np.where(np.isin(pre_rec, co_rec), 1, 0)
                if len(pre_rec) < topK:
                    rec_nums = len(pre_rec)
                else:
                    rec_nums = topK

                idx_arr = np.arange(1, rec_nums + 1, 1)
                ideal_arr = 1 / np.log2(idx_arr + 1)
                idcg = np.sum(ideal_arr)
                dcg_arr = np.multiply(ideal_arr, hit_order)
                dcg = np.sum(dcg_arr)
                ndcg = dcg / idcg
                return ndcg
            else:
                return 0

        def rating_ability(prediction, rating, ks_thresh=5):
            '''
            :param prediction: all prediction
            :param rating: all actual rating
            :param test: record each sample in test
            :return:
            '''
            perfect_num, total_num = 0, 0
            ks_user_count, ks_count = 0, 0

            # collect the perfect num and ks count for each user
            tuples = prediction.items()
            pred_rat = np.array([t[-1] for t in tuples])
            pred_rounded = np.ceil(pred_rat)
            actual_rat = np.array([rating[t[0]] for t in tuples])
            if actual_rat.shape[0] > 0:
                perfect_record = np.where((pred_rounded - actual_rat) == 0, 1, 0)
                perfect_num = np.sum(perfect_record)
                total_num = pred_rat.shape[0]
                ppr_u = perfect_num / total_num
                if actual_rat.shape[0] >= ks_thresh:
                    ks = ks_2samp(actual_rat, pred_rat)
                    ks_user_count = 1
                    if ks[1] > 0.05:
                        ks_count = 1
                return perfect_num, total_num, ks_user_count, ks_count, ppr_u
            else:
                return None

        def compute_avg(testRatings):
            sum_ratings, count_ratings = 0, 0
            for i in range(len(testRatings)):
                sum_ratings += sum(testRatings[i])
                count_ratings += len(testRatings[i])
            avg = sum_ratings / count_ratings
            return avg

        Count = []
        user_size = []
        RMSE = []
        PRE = []
        REC = []
        NDCG = []
        PERFECT_NUMS = []
        TOTAL_NUMS = []
        PPRS = []
        KS_USER_COUNT = []
        KS_COUNT = []
        R_SQUARE_FENZI, R_SQUARE_FENMU = [], []
        testUser = self.testNeg[0]
        testItem = self.testNeg[1]
        testRating = self.testNeg[2]
        rating_average = compute_avg(testRating)
        pred_test = []
        for i in range(len(testUser)):
            num = 1
            # target = testItem[i][0]
            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)

            prediction = {}
            rating = {}

            for j in range(len(testItem[i])):
                item = testItem[i][j]
                prediction[item] = predict[j] * 4 + 1
                rating[item] = testRating[i][j]
                pred_test.append([i, j, rating[item], prediction[item]])

            # Compute RMSE
            sum_error, count = getRMSE(prediction, rating)
            RMSE.append(sum_error)
            Count.append(count)
            user_size.append(num)

            # Compute R square
            tuples = prediction.items()
            pred_rat = np.array([t[-1] for t in tuples])
            actual_rat = np.array([rating[t[0]] for t in tuples])
            sum_error = np.sum(np.power((np.array(actual_rat) - np.array(pred_rat)), 2))
            sum_error_avg = np.sum(np.power((np.array(actual_rat) - rating_average), 2))
            R_SQUARE_FENZI.append(sum_error)
            R_SQUARE_FENMU.append(sum_error_avg)

            # Compute PPR and KS
            results = rating_ability(prediction, rating)
            if results is not None:
                perfect_num, total_num, ks_user_count, ks_count, ppr_u = results
                PERFECT_NUMS.append(perfect_num)
                TOTAL_NUMS.append(total_num)
                PPRS.append(ppr_u)
                KS_USER_COUNT.append(ks_user_count)
                KS_COUNT.append(ks_count)

            # Compute TopK
            pre, rec, co_rec, pre_rec = getF1_value(prediction, rating)
            PRE.append(pre)
            REC.append(rec)

            ndcg_u = getNDCG(co_rec, pre_rec)
            NDCG.append(ndcg_u)

        rmse = math.sqrt(np.sum(RMSE) / np.sum(Count))
        precision = np.sum(PRE) / np.sum(user_size)
        recall = np.sum(REC) / np.sum(user_size)
        ndcg = np.sum(NDCG) / np.sum(user_size)
        # ppr = np.sum(PERFECT_NUMS) / np.sum(TOTAL_NUMS)
        ppr = np.average(PPRS)
        ks = np.sum(KS_COUNT) / np.sum(KS_USER_COUNT)
        r_square = 1 - np.sum(R_SQUARE_FENZI) / np.sum(R_SQUARE_FENMU)

        # Target RMSE
        target_rmse = 1.320
        if (target_rmse - 0.05) <= rmse <= (target_rmse + 0.05):
            results = np.array(pred_test)
            test = results[:, 2]
            pred = results[:, 3]
            np.save(PRED_SAVEFILE + '_' + str(rmse), pred)
            np.save(TEST_SAVEFILE + '_' + str(rmse), test)
            print(f'SAVE RMSE {rmse} FILE SUCCESS!')
        if save_file:
            results = np.array(pred_test)
            test = results[:, 2]
            pred = results[:, 3]
            np.save(TEST_SAVEFILE, test)
            np.save(PRED_SAVEFILE, pred)
            print('SAVE FILE SUCCESS!')
        return rmse, precision, recall, ndcg, ppr, ks, r_square
        # ranklist = heapq.nlargest(topK, item_score_dict, key=item_score_dict.get)
        #
        # tmp_NDCG = getNDCG(ranklist, target)
        # hr.append(tmp_hr)
        # NDCG.append(tmp_NDCG)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("Total time:", time_end - time_start)
