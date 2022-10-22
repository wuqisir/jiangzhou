'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import utility.metrics as metrics
from utility.parser import args
from utility.load_data import *
import multiprocessing
import heapq
import pandas as pd
import math
from utility.helper import set_seed
import torch.nn as nn

set_seed(2021)
cores = multiprocessing.cpu_count() // 2

Ks = eval(args.Ks)

data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)

USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]
    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)
    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(u, user_pos_test, rating, auc, Ks):
    precision, recall, ndcg, hit_ratio, f1 = [], [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(user_pos_test, rating, K))
        recall.append(metrics.recall_at_k(user_pos_test, rating, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(user_pos_test, rating, K))
        hit_ratio.append(0)
        f1.append(cal_f1(u, user_pos_test, rating, K, len(user_pos_test)))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'f1': np.array(f1)}

def cal_f1(u, user_pos_item, rating, K, item_num):
    recall = metrics.recall_at_k(user_pos_item, rating, K, item_num)
    precision = metrics.precision_at_k(user_pos_item, rating, K)
    if (precision + recall) == 0:
        return 0.
    else:
        return 2*precision*recall/(precision + recall)

def test_one_user(x):
    # user u's ratings for user u
    rating = (x[0] * 4) + 1
    # rating = x[0]

    #uid
    u = x[1]
    tmp = np.load('D:\\test_matrix\\' + str(u) + '.npy')

    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]
    for id in user_pos_test:
        if tmp[id] <= 3:
            remove_id = [id]
            user_pos_test = list(set(user_pos_test) - set(remove_id))

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

#     print('test_items', len(test_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)
        
#     print(len(r))
    return get_performance(u, user_pos_test, rating, auc, Ks)

# cdf, rmse, ppr
def cdf_rmse_ppr(rate_batch, user_batch):
    rate_batch = (rate_batch * 4) + 1
    rmse = 0
    perfect_num = 0
    rate_batch_rounded = np.ceil(rate_batch)
    error_list = np.linspace(0, 5, 30).tolist()
    del error_list[0]
    dict_erro = dict.fromkeys(error_list, 0)
    for i in range(len(user_batch)):
        tmp = np.load('D:\\test_matrix\\' + str(user_batch[i]) + '.npy')
        tmp_cdf = tmp - rate_batch[i]
        item = data_generator.test_set[user_batch[i]]
        for id in item:
            rmse += tmp_cdf[id] **2
            if rate_batch_rounded[i, id] == tmp[id]:
                perfect_num += 1
            for key in dict_erro.keys():
                if tmp_cdf[id] < key:
                    dict_erro[key] += 1

    return dict_erro, rmse, perfect_num

# for NCF batch_test_flag=true
def test(model, users_to_test, drop_flag=False, batch_test_flag=True):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'f1': np.zeros(len(Ks))}

    error_list = np.linspace(0, 5, 30).tolist()
    del error_list[0]
    dict_cdf = dict.fromkeys(error_list, 0)
    rmse = 0
    # caculate ppr
    perfect_num = 0

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    all_result = np.zeros((n_test_users, ITEM_NUM))
    all_user = np.zeros(n_test_users)

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = ITEM_NUM // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))
            rate_batch1 = np.zeros(shape=(len(user_batch), ITEM_NUM))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)
                # batch items
                item_batch = range(i_start, i_end)
                # batch u_i rating
                i_rate_batch = model(user_batch,item_batch).detach().cpu() # forward
                # connect batch
                # i_rate_batch = model(user_batch,item_batch).detach().numpy()
                # c = i_rate_batch
                i_rate_batch = nn.Sigmoid()(i_rate_batch)
                rate_batch[:, i_start: i_end] = i_rate_batch
                # rate_batch1[:, i_start: i_end] = c
                i_count += i_rate_batch.shape[1]

            assert i_count == ITEM_NUM
        else:
            # all-item test
            item_batch = range(ITEM_NUM)
            rate_batch = model(user_batch,item_batch).detach().cpu().numpy() # forward
        
        user_batch_rating_uid = zip(rate_batch, user_batch)
        tmp_cdf, tmp_rmse, tmp_ppr = cdf_rmse_ppr(rate_batch, user_batch)
        all_result[user_batch] = rate_batch
        all_user[user_batch] = user_batch
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for key in tmp_cdf.keys():
            dict_cdf[key] += tmp_cdf[key]
        rmse += tmp_rmse
        perfect_num += tmp_ppr

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['f1'] += re['f1']/n_test_users

    # top_k, recall, precision, f1_score, hr, ndcg, ppr, ks = recall_precision_f1_hr(all_result, data_generator.test_matrix, top_k=5)
    assert count == n_test_users
    pool.close()
    test_matrix = np.load(f"./dataset/{args.dataset}/{args.dataset}_test.npy")
    rate_num = len((np.nonzero(test_matrix))[0])
    for key in dict_cdf.keys():
        dict_cdf[key] /= rate_num
    rmse = math.sqrt(rmse / rate_num)
    ppr = perfect_num / rate_num
    print(all_result)

    return result, dict_cdf, rmse, ppr, all_result, all_user
