from utils.LoadData import DataSet
from als import ALS
import numpy as np
import pandas as pd


# save results
def saverecording(record, file, columns):
    pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)


if __name__ == '__main__':
    # trainpath = '../data/DM/trainset(DM).xlsx'
    # testpath = '../data/DM/testset(DM).xlsx'
    # trainpath = '../data/IV/trainset(IV).xlsx'
    # testpath = '../data/IV/testset(IV).xlsx'
    # trainpath = '../data/ML100K/trainset.txt'
    # testpath = '../data/ML100K/testset.txt'
    trainpath = '../data/AA/trainset(AA).xlsx'
    testpath = '../data/AA/testset(AA).xlsx'
    result_file = '{}_ALS_result.csv'.format(trainpath)
    result_file_rmse = '{}_ALS_rmse_result.csv'.format(trainpath)
    result_file_dimension = '{}_ALS_result_dimension.csv'.format(trainpath)
    normalization = False
    ds = DataSet(trainpath, testpath)
    TrainSetRatings, _ = ds.list_to_matrix(ds.trainset, ds.maxu, ds.maxi)
    TestSetRatings, _ = ds.list_to_matrix(ds.testset, ds.maxu, ds.maxi)
    if normalization:
        TrainSetRatings = ds.MinMaxScaler(TrainSetRatings)
    record = []
    topk = 5
    # for f in [5, 10, 20, 30, 40, 50]:
    for f in [50]:
        print('f:{}'.format(f))
        print('=' * 20)
        for reg in [0.01]:
            print('reg:{}'.format(reg))
            print('=' * 20)
            als = ALS()
            als.get_params(ds.datamin, ds.datamax)
            np.random.seed(2)
            regP, regQ = reg, reg
            # P0, Q0 = als.initPQ(TrainSetRatings, f)
            P0, Q0 = als.initPQGaussian(TrainSetRatings, f)
            # P0, Q0 = np.load('./data/backup/initP.npy'), np.load('./data/backup/initQ.npy')
            print(P0[:5])
            print(Q0[:5])
            iterarray = np.arange(1000)
            P, Q, *results = als.ALS(TrainSetRatings, P0, Q0, iterarray, f, regP, regQ, TestSetRatings, topk=5,
                                     clip=False,
                                     normlization=normalization,
                                     verbose=True,
                                     save_file=result_file,
                                     save_file_rmse=result_file_rmse)
        record.append(results)
    saverecording(record, result_file_dimension, ['f', 'RMSE', 'rec@{}'.format(topk), 'pre@{}'.format(topk),
                                                  'f1@{}'.format(topk), 'hr', 'ndcg@{}'.format(topk), 'ppr', 'ks',
                                                  'r_square'])
