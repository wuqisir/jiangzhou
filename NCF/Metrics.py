import numpy as np
import math
import pandas as pd
from scipy.stats import ks_2samp
from some_stuff import plot_denshist, quicklook


# def mrr(gt_item, pred_items):
#     if gt_item in pred_items:
#         index = np.where(pred_items == gt_item)[0][0]
#         return np.reciprocal(float(index + 1)) #返回倒数
#     else:
#         return 0
#
# def hit(gt_item, pred_items):
#     if gt_item in pred_items:
#         return 1
#     return 0
def pred_value(predict):
    prediction = []
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            p = predict[i][j] * 4 + 1
            prediction.append(p)
    return prediction


def rmse(prediction, rating):
    # np.save('pred_array_NCF', np.array(prediction))
    # np.save('test_array_NCF', np.array(rating))
    sum_error = 0
    count = 0
    for i in range(len(prediction)):
        error = math.fabs(prediction[i] - rating[i])
        sum_error += error * error
        count += 1
    RMSE = math.sqrt(sum_error / count)
    return RMSE


def rounded_rmse(prediction, rating):
    sum_error = 0
    count = 0
    for i in range(len(prediction)):
        # error = math.fabs(np.round(prediction[i]) - rating[i])
        error = math.fabs(np.ceil(prediction[i]) - rating[i])
        sum_error += error * error
        count += 1
    RMSE = math.sqrt(sum_error / count)
    return RMSE


def R_square(predict_array, test_array):
    '''
    :param predict_array: nonzero ndarray
    :param test_array:
    :return:
    '''
    predict_array = np.array(predict_array)
    test_array = np.array(test_array)
    difference = np.subtract(predict_array, test_array)
    diff_square = np.power(difference, 2)
    sum_diff_square = np.sum(diff_square)

    rating_avg = np.average(test_array)
    difference2 = np.subtract(test_array, rating_avg)
    sum_diff_square2 = np.sum(np.power(difference2, 2))

    return 1 - sum_diff_square / (sum_diff_square2 + 1e-7)


def rating_ability(prediction, rating, test, ks_thresh=5):
    '''
    :param prediction: all prediction
    :param rating: all actual rating
    :param test: record each sample in test
    :return:
    '''
    actual_rec = {}
    predict_rec = {}
    perfect_nums, total_nums = [], []
    pprs = []
    ks_user_count, ks_count = 0, 0

    # form the user_ratings_dict and user_predictions_dict
    for i in range(len(test)):
        u = test['user'][i]  # get user id
        t = test['item'][i]  # get item id
        if rating[i] == 0: continue
        # actual_rec[u].setdefault(t, rating[i])
        if u in actual_rec:
            actual_rec[u].append(rating[i])
            predict_rec[u].append(prediction[i])
        else:
            actual_rec.setdefault(u, [rating[i]])
            predict_rec.setdefault(u, [prediction[i]])

    # collect the perfect num and ks count for each user
    for u in predict_rec.keys():
        pred_rat = np.array(predict_rec[u])
        pred_rounded = np.ceil(pred_rat)
        # pred_rounded = np.round(pred_rat)
        # pred_rounded = np.clip(pred_rounded, 0, 5)
        actual_rat = np.array(actual_rec[u])
        if actual_rat.shape[0] > 0:
            perfect_record = np.where((pred_rounded - actual_rat) == 0, 1, 0)
            perfect_num = np.sum(perfect_record)
            total_num = actual_rat.shape[0]
            ppr_u = perfect_num / total_num
            pprs.append(ppr_u)
            total_nums.append(total_num)
            perfect_nums.append(perfect_num)
        if actual_rat.shape[0] >= ks_thresh:
            ks = ks_2samp(actual_rat, pred_rat)
            ks_user_count += 1
            if ks[1] > 0.05:
                ks_count += 1
    # PPR for all
    # valid_test_ratings = np.sum(total_nums)
    # perfect_ratings = np.sum(perfect_nums)
    # total_nums_look = quicklook(total_nums, [0, 10, 100, 1000])
    # perfect_nums_look = quicklook(perfect_nums, [0, 10, 100, 1000])
    # print(f'total_nums_look:{total_nums_look}')
    # print(f'perfect_nums_look:{perfect_nums_look}')
    # ppr = np.sum(perfect_nums) / np.sum(total_nums)

    # PPR user level
    ppr = np.average(pprs)
    ks = ks_count / ks_user_count
    return ppr, ks


# def f1_value(prediction, rating, test, k):
#     actual_rec = {}
#     predict_rec = {}
#     invalid_recs, unpredictable = 0, 0
#     pre = []
#     rec = []
#     ndcg = []
#     for i in range(len(test)):
#         u = test['user'][i]
#         t = test['item'][i]
#         actual_rec.setdefault(u, {})
#         predict_rec.setdefault(u, {})
#         if rating[i] > 3:  # rating[i]
#             actual_rec[u].setdefault(t, rating[i])
#         if prediction[i] > 3:
#             predict_rec[u].setdefault(t, prediction[i])
#
#     for user in predict_rec.keys():
#         pred_top = sorted(predict_rec[user].items(), key=lambda x: x[1], reverse=True)[0:k]
#         count = 0.0
#         num = 0.0
#
#         pre_rec = []
#         co_rec = []
#         for item, rating in pred_top:
#             num += 1
#             pre_rec.append(item)
#             if item in actual_rec[user].keys():
#                 count += 1
#                 co_rec.append(item)
#
#         if count > 0:
#             hit_order = np.where(np.isin(pre_rec, co_rec), 1, 0)
#             if num < k:
#                 rec_nums = num
#             else:
#                 rec_nums = k
#             pre.append(count / rec_nums)
#             rec.append(count / len(actual_rec[user].keys()))
#
#             idx_arr = np.arange(1, rec_nums + 1, 1)
#             ideal_arr = 1 / np.log2(idx_arr + 1)
#             idcg = np.sum(ideal_arr)
#             dcg_arr = np.multiply(ideal_arr, hit_order)
#             dcg = np.sum(dcg_arr)
#             ndcg_u = dcg / idcg
#             ndcg.append(ndcg_u)
#
#         else:
#             pre.append(0)
#             rec.append(0)
#             ndcg.append(0)
#         if count == 0:
#             invalid_recs += 1
#         if len(actual_rec[user].keys()) == 0:
#             unpredictable += 1
#     print(f"invalid_recs:{invalid_recs}")
#     print(f"unpredictables:{unpredictable}")
#     print(f"per_sum:{sum(pre)}; pre_len:{len(pre)}; nonzero_nums:{len(np.nonzero(pre)[0])}")
#     PRE = sum(pre) / len(pre)
#     REC = sum(rec) / len(rec)
#     F1 = (2.0 * PRE * REC) / (PRE + REC)
#     NDCG = sum(ndcg) / len(ndcg)
#
#     return PRE, REC, F1, NDCG

def f1_value(prediction, rating, test, k):
    actual_rec = {}
    predict_rec = {}
    pre = []
    rec = []
    ndcg = []
    hr = []
    #f = open('result.txt', 'w', encoding='utf-8')
    for i in range(len(test)):
        u = test['user'][i]
        t = test['item'][i]
        actual_rec.setdefault(u, {})
        predict_rec.setdefault(u, {})
        if rating[i] >= 3:
            actual_rec[u].setdefault(t, rating[i])
        if prediction[i] >= 3:
            predict_rec[u].setdefault(t, prediction[i])

        #这部分是给实际列表和推荐列表补充5个负样本，这些项目不用于推荐
        actual_rec[u].setdefault(-1, -1)
        actual_rec[u].setdefault(-2, -2)
        actual_rec[u].setdefault(-3, -3)
        actual_rec[u].setdefault(-4, -4)
        actual_rec[u].setdefault(-5, -5)

        predict_rec[u].setdefault(-6, -6)
        predict_rec[u].setdefault(-7, -7)
        predict_rec[u].setdefault(-8, -8)
        predict_rec[u].setdefault(-9, -9)
        predict_rec[u].setdefault(-10, -10)

    for user in actual_rec.keys():
        if len(actual_rec[user]) > 0:
            actual_top = sorted(actual_rec[user].items(), key=lambda x: x[1], reverse=True)[0:k]
            #f.write(str(user)+str(actual_top)+'\t')
            if len(predict_rec[user]) > 0:
                pred_top = sorted(predict_rec[user].items(), key=lambda x: x[1], reverse=True)[0:k]
                #f.write(str(user)+str(pred_top)+'\n')
                count = 0.0
                num = 0.0
                pre_rec = []
                act_rec = []
                co_rec = []

                for item, rating in actual_top:
                    act_rec.append(item)

                for item, rating in pred_top:
                    num += 1
                    pre_rec.append(item)
                        # pre,rec
                        # if item in actual_rec.keys():
                        #     count += 1
                        #     co_rec.append(item)

                        # hr:
                    if item in act_rec:
                        count += 1
                        co_rec.append(item)

                if count > 0:
                    hit_order = np.where(np.isin(pre_rec, co_rec), 1, 0)#k维
                    rec_nums = k
                    pre.append(count / rec_nums)
                    rec.append(count / k)

                    hr.append(count / k)

                    idx_arr = np.arange(1, rec_nums + 1, 1)
                    ideal_arr = 1 / np.log2(idx_arr + 1)
                    idcg = np.sum(ideal_arr)
                    dcg_arr = np.multiply(ideal_arr, hit_order)
                    dcg = np.sum(dcg_arr)
                    ndcg_u = dcg / idcg
                    ndcg.append(ndcg_u)
                else:
                    pre.append(count)
                    rec.append(count)
                    ndcg.append(count)
                    hr.append(count)
            else:
                pre.append(0.0)
                rec.append(0.0)
                ndcg.append(0.0)
                hr.append(0.0)
        else:
            continue

    PRE = sum(pre) / len(pre)
    REC = sum(rec) / len(rec)
    F1 = (2.0 * PRE * REC) / (PRE + REC)
    NDCG = sum(ndcg) / len(ndcg)
    HR = sum(hr) / len(hr)

    return PRE, REC, F1, HR, NDCG

#
# def ndcg(test, prediction):
#     if gt_item in pred_items:
#         index = np.where(pred_items == gt_item)[0][0]
#         return np.reciprocal(np.log2(index + 2))
#     return 0
