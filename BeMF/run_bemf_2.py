import torch
import utils.LoadData as LoadData
import pandas as pd
from bemf_2 import (BaseModule, BasePipeline)
import numpy as np


# save results
def saverecording(record, file, columns):
    pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)


def bemf2(train_rat_path, test_rat_path, result_path=None, result_path_rmse=None, result_file_dimension=None):
    train_rat_mat, test_rat_mat, \
    train_u_items_dict, test_u_items_dict, train_u_avg, _ = LoadData.get_spmat_train_test(train_rat_path,
                                                                                          test_rat_path,
                                                                                          normalization=False,
                                                                                          u_avg=False)
    best_rmse = 1000
    topk = 5
    best_parmas = []
    record = []
    for reg in [0.01]:
        for lr in [0.02]:
            print('params: lr:{}, reg:{}'.format(lr, reg))
            print('=' * 50)
            # for f in [5, 10, 20, 30, 40, 50]:
            for f in [10]:
                print('f:{}'.format(f))
                print('=' * 10)
                train_mat_iterator = LoadData.get_rel_mat_dict(train_rat_mat, np.array([1, 2, 3, 4, 5]))
                pipeline = BasePipeline(train_mat_iterator,
                                        test_rat_mat,
                                        train_u_avg_dict=train_u_avg,
                                        rate_range=np.array([1, 2, 3, 4, 5]),
                                        n_factors=f,
                                        batch_size=1024,
                                        lr=lr,
                                        n_epochs=20,
                                        topk=5,
                                        reg_user=reg,
                                        reg_item_rat=reg,
                                        data_normalization=False)
                results = pipeline.fit(save_file=result_path, save_file_rmse=result_path_rmse, rank_metric=True)
                record.append(results)
                cur_rmse = results[0]
                if cur_rmse < best_rmse:
                    best_rmse = cur_rmse
                    best_params = [lr, reg]
        saverecording(record, result_file_dimension, ['f', 'RMSE', 'rec@{}'.format(topk), 'pre@{}'.format(topk),
                                                      'f1@{}'.format(topk), 'hr', 'ndcg@{}'.format(topk)])
    print('best_rmse:{}'.format(best_rmse))
    print('best params:{}'.format(best_params))


if __name__ == '__main__':
    interaction_type = 'implicit'
    train_rat_path = './data/IV/trainset(IV).xlsx'
    test_rat_path = './data/IV/testset(IV).xlsx'
    # train_rat_path = '../data/DM/trainset(DM).xlsx'
    # test_rat_path = '../data/DM/testset(DM).xlsx'
    result_file = '{}_BeMF_result.csv'.format(train_rat_path)
    result_file_rmse = '{}_BeMF_rmse_result.csv'.format(train_rat_path)
    result_file_dimension = '{}_BeMF_result_dimension.csv'.format(train_rat_path)
    bemf2(train_rat_path, test_rat_path, result_file, result_file_rmse, result_file_dimension)
