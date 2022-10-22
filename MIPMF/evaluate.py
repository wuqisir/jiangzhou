import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from utils.Others import plot_denshist
from utils.some_stuff import plot_denshist, quicklook


def dcg(hit_order, rec_nums):
    if len(hit_order) >= 1:
        idx_arr = np.arange(1, rec_nums + 1, 1)
        ideal_arr = 1 / np.log2(idx_arr + 1)
        idcg = np.sum(ideal_arr)
        dcg_arr = np.multiply(ideal_arr, hit_order)
        dcg = np.sum(dcg_arr)
        ndcg_u = dcg / idcg
        return ndcg_u
    else:
        return 0


def compute_rmse(predict_matrix, test_mat, clip=True, rounded=False):
    predictions = predict_matrix[predict_matrix.nonzero()]
    actual_ratings = test_mat[test_mat.nonzero()]
    if clip:
        predictions = np.clip(predictions, 1, 5)
    if rounded:
        predictions = np.round(predictions)
        # predictions = np.ceil(predictions)
    temp = np.subtract(predictions, actual_ratings)
    temp1 = np.power(temp, 2)
    temp2 = np.sum(temp1)
    num = actual_ratings.shape[0]
    temp3 = temp2 / num
    RMSE = np.sqrt(temp3)
    return RMSE


def R_square(predict_array, test_array):
    '''
    :param predict_array: nonzero ndarray
    :param test_array:
    :return:
    '''
    difference = np.subtract(predict_array, test_array)
    diff_square = np.power(difference, 2)
    sum_diff_square = np.sum(diff_square)

    rating_avg = np.average(test_array)
    difference2 = np.subtract(test_array, rating_avg)
    sum_diff_square2 = np.sum(np.power(difference2, 2))

    return 1 - sum_diff_square / (sum_diff_square2 + 1e-7)


def recall_precision_f1_hr(predict_matrix, predict_matrix_rel, test_mat, train_u_avg, top_k, rec_thresh=3, ks_thresh=5):
    '''
    compute the indicators in the name of this function
    :param predict_matrix:
    :param predict_matrix_rel: reliable predicting matrix
    :param test_mat: ndarray test matrix
    :param train_u_avg: dict
    :param top_k:
    :param rec_thresh:
    :return:
    '''
    # np.save('pred_mat_MIPMF', predict_matrix)
    # np.save('test_mat_MIPMF', test_mat)
    recalls, precisions = [], []
    ndcgs = []
    actual_items_list, rec_items_list = [], []
    total_hit, total_actual_items = 0, 0
    total_nums, perfect_nums = [], []
    pprs = []
    users = 0
    unpredictable = 0
    invalid_rec = 0
    ks_user_count, ks_count = 0, 0
    actual_nums_list = []
    r_square_list = []
    rmse = compute_rmse(predict_matrix, test_mat, clip=True, rounded=True)
    print(f'original rmse:{compute_rmse(predict_matrix, test_mat, clip=False, rounded=False)}')
    print(f'cliped rmse:{compute_rmse(predict_matrix, test_mat, clip=True, rounded=False)}')
    print('rounded rmse:{}'.format(rmse))
    r_square = R_square(predict_matrix[predict_matrix.nonzero()], test_mat[test_mat.nonzero()])
    for u in range(test_mat.shape[0]):
        # TODO: combine the 2 lines below
        actual_vec = test_mat[u]
        actual_vec = np.array(actual_vec).squeeze()  # transfer the matrix to array
        actual_nums = len(actual_vec[actual_vec > 0])  # filter out the unpredictable entries
        actual_nums_list.append(actual_nums)
        rec_nums = top_k
        if actual_nums >= 1:

            # get the predict ratings
            pred_rat_vec = predict_matrix[u]
            pred_vec = pred_rat_vec
            # print(f'pred_rat_vec_id:{id(pred_rat_vec)}, pred_vec_id:{id(pred_vec)}')
            pred_vec = pd.Series(pred_vec)
            # print(f'pred_rat_vec_id:{id(pred_rat_vec)}, pred_vec_id:{id(pred_vec)}')
            pred_vec = pred_vec.sort_values(ascending=False)

            # # get the rounded ratings (ppr computing)
            test_ratings = actual_vec[actual_vec > 0]
            pred_ratings = pred_rat_vec[actual_vec > 0]
            pred_ratings_rounded = np.ceil(pred_ratings)
            # pred_ratings_rounded = np.round(pred_ratings)
            # pred_ratings_rounded = np.clip(pred_ratings_rounded, 1, 5)
            perfect_preds = np.where((pred_ratings_rounded - test_ratings) == 0, 1, 0)
            perfect_num = np.sum(perfect_preds)
            total_num = test_ratings.shape[0]
            ppr_u = perfect_num / total_num
            pprs.append(ppr_u)
            perfect_nums.append(perfect_num)
            total_nums.append(total_num)

            # get the KS
            if actual_nums >= ks_thresh:
                ks = ks_2samp(test_ratings, pred_ratings)  # not the rounded prediction
                ks_user_count += 1
                if ks[1] > 0.05:
                    ks_count += 1

            # get the predict reliability
            pred_rel = predict_matrix_rel[u]  # get the reliability
            pred_rat_rel = np.multiply(pred_rat_vec, pred_rel)  # correct the predictions with reliability
            pred_rat_rel = pd.Series(pred_rat_rel)

            # recommend based on the average value
            # avg_u = train_u_avg[u]
            # actual_items = np.where(actual_vec >= avg_u)[0]  # np.where return a tuple
            # rec_items = np.array(pred_vec[pred_vec >= avg_u][:rec_nums].index)

            # recommend based on the threshold value
            actual_items = np.where(actual_vec > rec_thresh)[0]  # np.where return a tuple
            rec_items = np.array(pred_vec[pred_vec > rec_thresh][:rec_nums].index)

            # get the recommending items
            rec_nums = len(rec_items)
            actual_items_list.append(len(actual_items))
            rec_items_list.append(rec_nums)

            # tackle the special cases
            if len(actual_items) == 0:  # unpredictable
                unpredictable += 1
                total_hit += 0
                total_actual_items += len(actual_items)
                recalls.append(0)
                precisions.append(0)
                ndcgs.append(0)
                continue
            if len(rec_items) == 0:  # invalid recommendations
                invalid_rec += 1
                total_hit += 0
                total_actual_items += len(actual_items)
                recalls.append(0)
                precisions.append(0)
                ndcgs.append(0)
                continue

            # evaluate the performance of recommendations
            hit_items = np.intersect1d(rec_items, actual_items)

            # compute the ndcg, only ndcg needs reoder
            rec_items_rel = pred_rat_rel[rec_items].sort_values(ascending=False).index  # reorder according to rel
            hit_order = np.where(np.isin(rec_items_rel, hit_items), 1, 0)
            ndcg_u = dcg(hit_order, rec_nums)  # compute the ndcg for single user u

            # compute the hit-based metrics
            hit = len(np.intersect1d(rec_items, actual_items))
            total_hit += hit
            total_actual_items += len(actual_items)
            recall = hit / len(actual_items)
            precision = hit / rec_nums
            # precision = hit / top_k
            recalls.append(recall)
            precisions.append(precision)
            ndcgs.append(ndcg_u)
            users += 1

    print('invalid recs:{}'.format(invalid_rec))
    print('unpredictable_users:{}'.format(unpredictable))

    # plot the frequency distribution
    # plot_denshist(actual_items_list, 'actual')
    # plot_denshist(rec_items_list, 'rec')
    # plot_denshist(actual_nums_list, 'actual_rates')

    recall = sum(recalls) / len(recalls)
    precision = sum(precisions) / len(precisions)
    ndcg = sum(ndcgs) / len(ndcgs)
    hr = total_hit / total_actual_items
    f1_score = 2 * recall * precision / (recall + precision + 1e-7)

    # Compute ppr on all test ratings
    # valid_test_ratings = np.sum(total_nums)
    # perfect_ratings = np.sum(perfect_nums)
    # total_nums_look = quicklook(total_nums, [0, 10, 100, 1000])
    # perfect_nums_look = quicklook(perfect_nums, [0, 10, 100, 1000])
    # print(f'total_nums_look:{total_nums_look}')
    # print(f'perfect_nums_look:{perfect_nums_look}')
    # ppr = sum(perfect_nums) / sum(total_nums)

    # Compute ppr user level
    ppr = np.average(pprs)
    ks = ks_count / ks_user_count
    # print('ks valid user nums:{}'.format(ks_user_count / len(total_nums)))
    print(f"user_nums:{len(pprs)}; ks_user_count:{ks_user_count}")
    return top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square
