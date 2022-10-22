import numpy as np
import pandas as pd
from sgd import Matrix_Factorization
from evaluate import recall_precision_f1_hr
from utils.Evaluation import revert_MinMaxScaler
from time import time

PRED_SAVEFILE = './pred_mat_ALS'
TEST_SAVEFILE = './test_mat_ALS'


class ALS(Matrix_Factorization):
    def __init__(self):
        super().__init__()
        self.rmse_results_recording = []
        self.results_recording = []

    def _eval_metrics(self, pred_matrix, test_ndmat, train_u_avg, topk=5, save_file=False):
        # save evaluating file
        if save_file:
            np.save(PRED_SAVEFILE, pred_matrix)
            np.save(TEST_SAVEFILE, test_ndmat)
            print('SAVE FILE SUCCESS!')

        # evaluating
        top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = recall_precision_f1_hr(pred_matrix,
                                                                                                 test_ndmat,
                                                                                                 train_u_avg,
                                                                                                 top_k=topk)

        return top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square

        # save results

    def saverecording(self, record, file, columns):
        pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)

    # ALS
    def ALS(self, R, P, Q, iter_array, f, regP, regQ, testR, bound=(-1, 1), topk=5, clip=False, normlization=True,
            verbose=True, save_file=None, save_file_rmse=None):
        if verbose:
            fvalue = self.train_verbose(R, P, Q, regP, regQ)
            rmse = self.test_verbose(testR, P, Q, self.datamin, self.datamax, revert_norm=normlization)
        for iter in range(1, iter_array[-1] + 2):
            count = 0
            s_time = time()
            for u in range(len(R)):
                Ru = R[u, :]
                Ru_hat = Ru[Ru.nonzero()]
                if len(Ru_hat) == 0:
                    continue
                Q_hat = Q[Ru.nonzero()[0]]
                Pu = np.linalg.solve(Q_hat.T @ Q_hat + regP * len(Ru_hat) * np.identity(f), Q_hat.T @ Ru_hat)
                if clip == 'clip':
                    Pu = np.clip(Pu, bound[0], bound[-1])
                elif np.linalg.norm(Pu) > 1 and clip == 'l2':
                    Pu = Pu * (1 / np.linalg.norm(Pu))
                    count += 1
                elif clip == 'None':
                    pass
                P[u] = Pu
            # print('iter: {} clips: {}'.format(iter, count))
            for i in range(len(R[0])):
                Ri = R[:, i]
                Ri_hat = Ri[Ri.nonzero()]
                if len(Ri_hat) == 0:
                    continue
                P_hat = P[Ri.nonzero()[0]]
                Qi = np.linalg.solve(P_hat.T @ P_hat + regQ * len(Ri_hat) * np.identity(f), P_hat.T @ Ri_hat)
                # Qi = np.clip(Qi, bound[0], bound[-1])
                Q[i] = Qi
            print("training time :{}".format(time() - s_time))
            if verbose:
                fvalue = self.train_verbose(R, P, Q, regP, regQ)
                rmse = self.test_verbose(testR, P, Q, self.datamin, self.datamax, revert_norm=normlization)
                print(f'epoch:{iter}, rmse:{rmse}')
            if iter == iter_array[-1]:
                self.result_f = fvalue
                self.result_rmse = rmse
            if topk:
                pred_mat = P @ Q.T
                if normlization:
                    pred_mat = revert_MinMaxScaler(pred_mat, self.datamin, self.datamax)
                    pred_mat = np.clip(pred_mat, self.datamin, self.datamax)
                target_rmse = 1.386
                if (target_rmse - 0.05) <= rmse <= (target_rmse + 0.05):
                    np.save(PRED_SAVEFILE + '_' + str(iter) + '_' + str(rmse), pred_mat)
                    np.save(TEST_SAVEFILE + '_' + str(iter) + '_' + str(rmse), testR)
                    print('SAVE FILE SUCCESS!')
                # filter out the unpredictable test ratings
                if (iter % 10000) == 0:
                    save_flag = True if iter == (iter_array[-1] + 1) else False
                    record_matrix = np.where(testR, 1, 0)
                    pred_mat = np.multiply(record_matrix, pred_mat)
                    train_u_avg = None
                    top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = self._eval_metrics(pred_mat,
                                                                                                         testR,
                                                                                                         train_u_avg,
                                                                                                         topk=topk,
                                                                                                         save_file=save_flag)
                    print(
                        'epoch:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{},ppr:{},ks:{},r_square:{}'.format(
                            iter,
                            top_k,
                            recall,
                            precision,
                            f1_score,
                            hr, ndcg, ppr, ks, r_square))
                    self.rmse_results_recording.append([iter, rmse])
                    self.results_recording.append([iter, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square])
        if save_file:
            self.saverecording(self.results_recording, save_file,
                               columns=['epoch', 'rec@{}'.format(topk), 'pre@{}'.format(topk),
                                        'f1@{}'.format(topk), 'hr', 'ndcg@{}'.format(topk), 'ppr', 'ks', 'r_square'])
        if save_file_rmse:
            self.saverecording(self.rmse_results_recording, save_file_rmse,
                               columns=['epoch', 'RMSE'])
        print('迭代完成！')
        return P, Q, f, rmse, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square
