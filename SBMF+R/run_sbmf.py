import torch
import utils.LoadData as LoadData
from sbmf import (BaseModule, BasePipeline)
import pandas as pd


# save results
def saverecording(record, file, columns):
    pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)


def multiple(train_rat_path, test_rat_path, sentiment_path, weight_path,
             result_path=None,
             result_path_rmse=None,
             result_file_dimension=None):
    print('load rat mat')
    train_rat_mat, test_rat_mat, \
    train_u_items_dict, test_u_items_dict, train_u_avg = LoadData.get_spmat_train_test(train_rat_path,
                                                                                       test_rat_path,
                                                                                       normalization=True, u_avg=True)
    print('load sen mat')
    sen_mat, _, _, _, _ = LoadData.get_spmat_train_test(sentiment_path,
                                                        normalization=True)
    print('load weight mat')
    w_mat, _, _, _, _ = LoadData.get_spmat_train_test(weight_path,
                                                      binary=True,  # containing zero
                                                      normalization=False)
    best_rmse = 1000
    best_params = []
    record = []
    topk = 5
    for lr in [0.02]:
        for reg in [0.001]:
            print('params: lr:{}; reg:{}'.format(lr, reg))
            print('=' * 20)
            # for f in [5, 10, 20, 30, 40, 50]:
            for f in [10]:
                print('f:{}'.format(f))
                print('=' * 10)
                pipeline = BasePipeline(train_rat_mat,
                                        test_rat_mat,
                                        sen_mat,
                                        w_mat,
                                        test_u_items_dict=test_u_items_dict,
                                        train_u_avg_dict=train_u_avg,
                                        model=BaseModule,
                                        n_factors=f,
                                        batch_size=128,
                                        lr=lr,
                                        optimizer=torch.optim.SGD,
                                        n_epochs=100,
                                        topk=5,
                                        reg_user=reg,
                                        reg_item_rat=reg,
                                        data_normalization=True,
                                        init_type='guassian')
                results = pipeline.fit(save_file=result_path, save_file_rmse=result_path_rmse)
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
    # train_rat_path = './data/IV/trainset(IV).xlsx'
    # test_rat_path = './data/IV/testset(IV).xlsx'
    # sentiment_path = './data/IV/SBMF(IV_sentiment).xlsx'
    # weight_path = './data/IV/SBMF(IV_weight).xlsx'
    train_rat_path = '../data/DM/trainset(DM).xlsx'
    test_rat_path = '../data/DM/testset(DM).xlsx'
    sentiment_path = './data/DM/SBMF(DM_sentiment).xlsx'
    weight_path = './data/DM/SBMF(DM_weight).xlsx'
    result_file = '{}_SMBF_result.csv'.format(train_rat_path)
    result_file_rmse = '{}_SMBF_rmse_result.csv'.format(train_rat_path)
    result_file_dimension = '{}_SMBF_result_dimension.csv'.format(train_rat_path)
    # run
    multiple(train_rat_path, test_rat_path, sentiment_path, weight_path, result_file, result_file_rmse,
             result_file_dimension)
