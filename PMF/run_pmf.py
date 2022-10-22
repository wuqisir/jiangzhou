import torch
import utils.LoadData as LoadData
from pmf import (BaseModule, BasePipeline)
import scipy.sparse

def pmf(train_rat_path, test_rat_path, result_path=None, result_path_rmse=None, load_input=False):
    if load_input:
        train_rat_mat = scipy.sparse.load_npz(train_rat_path[:-5] + '.npz')  # txt
        test_rat_mat = scipy.sparse.load_npz(test_rat_path[:-5] + '.npz')
        print('rat mat is loaded!')
    else:
        train_rat_mat, test_rat_mat, \
        train_u_items_dict, test_u_items_dict, train_u_avg, _ = LoadData.get_spmat_train_test(train_rat_path,
                                                                                              test_rat_path,
                                                                                              normalization=False,
                                                                                              u_avg=False,
                                                                                              is_MT=True)

    pipeline = BasePipeline(train_rat_mat,
                            test_rat_mat,
                            train_u_avg_dict=None,
                            model=BaseModule,
                            n_factors=50,
                            batch_size=128,
                            lr=0.0025,
                            optimizer=torch.optim.SGD,
                            n_epochs=200,
                            topk=5,
                            reg_user=0.01,
                            reg_item_rat=0.01,
                            data_normalization=False,
                            init_type='guassian',
                            random_seed=0)
    pipeline.fit(save_file=result_path, save_file_rmse=result_path_rmse)


if __name__ == '__main__':
    interaction_type = 'implicit'
    # train_rat_path = '../data/IV/trainset(IV).xlsx'
    # test_rat_path = '../data/IV/testset(IV).xlsx'
    # train_rat_path = '../data/DM/trainset(DM).xlsx'
    # test_rat_path = '../data/DM/testset(DM).xlsx'
    # train_rat_path = '../data/AA/trainset(AA).xlsx'
    # test_rat_path = '../data/AA/testset(AA).xlsx'
    train_rat_path = '../data/MT/trainset(MT).txt'
    test_rat_path = '../data/MT/testset(MT).txt'
    result_file = '{}_PMF_result.csv'.format(train_rat_path)
    result_file_rmse = '{}_PMF_rmse_result.csv'.format(train_rat_path)
    pmf(train_rat_path, test_rat_path, result_file, result_file_rmse, load_input=False)
