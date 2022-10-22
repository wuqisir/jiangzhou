import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy
import math
from time import time
from tqdm import tqdm
from utils.Others import regularization, revert_MinMaxScaler
from evaluate import recall_precision_f1_hr

PRED_SAVEFILE = './pred_mat_PMF'
TEST_SAVEFILE = './test_mat_PMF'


class Interactions(data.Dataset):
    def __init__(self, ratmat):
        self.ratmat = ratmat.astype(np.float32).tocoo()
        self.n_users = self.ratmat.shape[0]
        self.n_items = self.ratmat.shape[1]

    def __getitem__(self, index):
        row = self.ratmat.row[index]  # different from getrow(index)
        col = self.ratmat.col[index]
        ratval = self.ratmat.data[index]
        return (row, col), ratval

    def __len__(self):  # when you use class as a function and the method you call is __call__()
        return self.ratmat.nnz


class Interactions_test(data.Dataset):
    def __init__(self, ratmat):
        self.ratmat = ratmat.astype(np.float32).tocoo()
        self.n_users = self.ratmat.shape[0]
        self.n_items = self.ratmat.shape[1]

    def __getitem__(self, index):
        row = self.ratmat.row[index]
        col = self.ratmat.col[index]
        ratval = self.ratmat.data[index]
        return (row, col), ratval

    def __len__(self):  # when you use class as a function and the method you call is __call__()
        return self.ratmat.nnz


def initPQ(n_users, n_items, n_factors):
    P = []
    Q = []
    for u in range(n_users):
        P.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    for i in range(n_items):
        Q.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    P = np.array(P)
    Q = np.array(Q)
    return P, Q


def initPQGaussian(n_user, n_item, n_factors):
    P = 0.1 * np.random.randn(n_user, n_factors)  # standard normal distribution
    Q = 0.1 * np.random.randn(n_item, n_factors)
    return P, Q


class BaseModule(nn.Module):
    # TODO: investigate the difference of the train RMSE
    # TODO: find a natural way to set the initialization of the embedding
    # TODO: investigate the gradient
    def __init__(self,
                 n_users,
                 n_items,
                 initP,
                 initQ,
                 n_factors=40,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 sparse=False):
        super(BaseModule, self).__init__()  # inherit the vars and functions in the super class
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_embedding = nn.Embedding(self.n_users, self.n_factors, _weight=initP, sparse=sparse)
        self.item_rat_embedding = nn.Embedding(self.n_items, self.n_factors, _weight=initQ, sparse=sparse)
        self.sparse = sparse
        self.reg_user = reg_user
        self.reg_item_rat = reg_item_rat
        self.is_MT = True

    def loss(self, data):
        (rows, cols), ratval = data

        # derive the rating embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)

        # derive the regularization
        ues_reg = regularization(ues, self.reg_user)  # calculate the regularization of P
        uis_rat_reg = regularization(uis_rat, self.reg_item_rat)  # calculate the regularization of Q

        # preds for rating
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)

        # calculate loss
        loss_rat = (preds_rat.squeeze() - ratval).pow(2).sum() + ues_reg + uis_rat_reg
        return loss_rat

    def forward(self, data):
        (rows, cols), ratval = data

        # derive the rating embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)

        # derive the regularization
        ues_reg = regularization(ues, self.reg_user)  # calculate the regularization of P
        uis_rat_reg = regularization(uis_rat, self.reg_item_rat)  # calculate the regularization of Q

        # preds for rating
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)

        # calculate loss
        # loss_rat = (preds_rat.squeeze() - ratval).pow(2).sum() + ues_reg + uis_rat_reg
        return preds_rat, ues_reg, uis_rat_reg

    def loss_rat_(self, data, data_normalization=False, datamin=1, datamax=5, clip=False):
        (rows, cols), ratval = data
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)
        if data_normalization:
            preds_rat = revert_MinMaxScaler(preds_rat, datamin, datamax)
        if clip:
            preds_rat = torch.clamp(preds_rat, datamin, datamax)
        loss_rat = (preds_rat.squeeze() - ratval).pow(2).sum()
        return loss_rat

    def predict(self, data_normalization=False, datamin=1, datamax=5, clip=False):
        P = self.user_embedding.weight.data
        Q = self.item_rat_embedding.weight.data
        if self.is_MT:
            P = P.detach().cpu().numpy()
            Q = Q.detach().cpu().numpy()
            pred_matrix = np.matmul(P, Q.T)
            if data_normalization:
                pred_matrix = revert_MinMaxScaler(pred_matrix, datamin, datamax)
            if clip:
                pred_matrix = np.clip(pred_matrix, datamin, datamax)
        else:
            pred_matrix = torch.mm(P, Q.t())
            if data_normalization:
                pred_matrix = revert_MinMaxScaler(pred_matrix, datamin, datamax)
            if clip:
                pred_matrix = torch.clamp(pred_matrix, datamin, datamax)
        return pred_matrix


class BasePipeline:
    def __init__(self,
                 train_rat_mat,
                 test_rat_mat,
                 model,
                 n_factors,
                 batch_size,
                 lr,
                 train_u_avg_dict,
                 sparse=False,
                 optimizer=torch.optim.SGD,
                 loss_function=nn.MSELoss(reduce=True, size_average=False),
                 n_epochs=10,
                 topk=5,
                 train_interaction_class=Interactions,
                 test_interaction_class=Interactions_test,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 data_normalization=False,
                 init_type='guassian',
                 random_seed=0):
        self.train_rat = train_rat_mat
        self.test_rat = test_rat_mat
        self.train_u_avg_dict = train_u_avg_dict
        # define the train_data
        self.train_loader = data.DataLoader(train_interaction_class(train_rat_mat),
                                            batch_size=batch_size,
                                            shuffle=True)
        self.test_loader = data.DataLoader(test_interaction_class(test_rat_mat), batch_size=batch_size,
                                           shuffle=True)
        # define some basic params
        self.n_users = self.train_rat.shape[0]
        self.n_items = self.train_rat.shape[1]
        self.n_factors = n_factors
        self.reg_user = reg_user
        self.reg_item_rat = reg_item_rat
        self.is_MT=True

        # set the initial latent matrix
        np.random.seed(random_seed)
        if init_type == 'guassian':
            initP, initQ = initPQGaussian(self.n_users, self.n_items, self.n_factors)
        elif init_type == 'original':
            initP, initQ = initPQ(self.n_users, self.n_items, self.n_factors)
        elif init_type == 'saved':
            initP = np.load('./data/backup/initP.npy')
            initQ = np.load('./data/backup/initQ.npy')
            print(initP[:5])
            print(initQ[:5])
        else:
            raise ValueError('invalid init type!')
        initP = torch.from_numpy(initP)
        initQ = torch.from_numpy(initQ)
        initP = initP.float()
        initQ = initQ.float()

        # define the optimization params
        self.batch_size = batch_size
        self.lr = lr
        self.loss_function = loss_function  # define the loss function type
        self.n_epochs = n_epochs
        self.model = model(self.n_users,
                           self.n_items,
                           initP,
                           initQ,
                           n_factors=self.n_factors,
                           sparse=sparse,
                           reg_user=self.reg_user,
                           reg_item_rat=self.reg_item_rat)
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, momentum=0.8)
        self.topk = topk
        self.data_norm = data_normalization
        self.results_recording = []
        self.rmse_results_recording = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # self.gpu_ids = [0, 1, 2]
        # self.gpu_ids = 1
        self.model.to(self.device)

    # save results
    def saverecording(self, record, file, columns):
        pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)

    def fit(self, save_file=None, save_file_rmse=None):
        for epoch in range(1, self.n_epochs + 1):
            s_time = time()
            train_loss = self._fit_epoch(epoch)
            e_time = time()
            row = 'Epoch: {0:^3}  time:{1:^10.3f} train:{2:^10.5f}'.format(epoch, e_time - s_time, train_loss)
            if self.test_rat is not None:
                test_rmse = self._validation_loss()
                row += 'val: {0:^10.5f}'.format(test_rmse)
                self.rmse_results_recording.append([epoch, test_rmse.item()])
                # if epoch % 3 == 0:
                # if (0 < (1.386 - test_rmse) <= 0.05) or (epoch % 10 == 0):
                target_rmse = 1.3
                if (target_rmse - 0.05) <= test_rmse <= (target_rmse + 0.05):

                    # get the prediction matrix
                    pred_matrix = self.model.predict(data_normalization=self.data_norm, clip=True)
                    if not self.is_MT:
                        pred_matrix = pred_matrix.detach().cpu().numpy()

                    # get the test matrix
                    test_ndmat = self.test_rat.toarray()

                    # filter out the unpredictable test ratings
                    record_matrix = np.where(test_ndmat, 1, 0)
                    pred_matrix = np.multiply(record_matrix, pred_matrix)

                    # savefile
                    pred_array = pred_matrix[pred_matrix.nonzero()]
                    test_array = test_ndmat[test_ndmat.nonzero()]
                    # np.save(PRED_SAVEFILE + '_' + str(self.epoch) + '_' + str(self.rmse_test[-1]), pred_mat)
                    # np.save(TEST_SAVEFILE + '_' + str(self.epoch) + '_' + str(self.rmse_test[-1]), test_mat)
                    np.save(PRED_SAVEFILE + '_' + str(epoch) + '_' + str(test_rmse), pred_array)
                    np.save(TEST_SAVEFILE + '_' + str(epoch) + '_' + str(test_rmse), test_array)
                    print('SAVE FILE SUCCESS!')


                    # save file
                    # np.save(PRED_SAVEFILE + '_' + str(epoch), pred_matrix)
                    # np.save(TEST_SAVEFILE + '_' + str(epoch), test_ndmat)
                    # print('SAVE FILE SUCCESS!')
                if epoch % 10000 == 0:
                    save_flag = True if epoch == self.n_epochs else False
                    top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = self._eval_metrics(self.model,
                                                                                                         self.test_rat,
                                                                                                         self.train_u_avg_dict,
                                                                                                         topk=self.topk,
                                                                                                         save_file=save_flag)
                    print(
                        'e:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{}, ppr:{}, ks:{}, r_square{}'.format(
                            epoch, top_k,
                            recall,
                            precision,
                            f1_score,
                            hr, ndcg, ppr, ks, r_square))
                    self.results_recording.append([epoch, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square])
            print(row)
        if save_file:
            self.saverecording(self.results_recording, save_file,
                               columns=['epoch', 'rec@{}'.format(self.topk), 'pre@{}'.format(self.topk),
                                        'f1@{}'.format(self.topk), 'hr', 'ndcg@{}'.format(self.topk), 'ppr', 'ks'])
        if save_file_rmse:
            self.saverecording(self.rmse_results_recording, save_file_rmse,
                               columns=['epoch', 'RMSE'])

    def _fit_epoch(self, epoch=1):
        # if len(self.gpu_ids) > 0:
        #     model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
        # else:
        model = self.model
        model.train()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), ratval) in enumerate(self.train_loader):
            print('\r' + 'epoch:' + str(epoch) + str(batch_idx) + '/' + str(len(self.train_loader)), end='', flush=True)
            self.optimizer.zero_grad()  # set the gradient to zero

            row = row.long().to(self.device)  # long has the same smeaning of int
            col = col.long().to(self.device)
            ratval = ratval.float().to(self.device)

            # compute loss
            # loss = self.model.loss(((row, col), ratval))

            # compute  forward
            preds, ues_reg, uis_rat_reg = model(((row, col), ratval))
            loss = (preds.squeeze() - ratval).pow(2).sum() + ues_reg + uis_rat_reg
            loss.backward()

            # check the gradient
            # for param in self.model.named_parameters():
            #     if 'user' in param[0]:
            #         print(param[0], '---', param[1].grad[row], param[1].grad[row + 1])
            # print('==' * 12)

            self.optimizer.step()
            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
        total_loss /= self.train_rat.nnz
        return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long().to(self.device)
            col = col.long().to(self.device)
            val = val.float().to(self.device)
            loss = self.model.loss_rat_(((row, col), val), data_normalization=self.data_norm, clip=True)
            total_loss += loss.item()
        total_loss /= self.test_rat.nnz
        rmse = torch.sqrt(total_loss[0])
        return rmse

    def _eval_metrics(self, model, test_matrix, train_u_avg, topk=5, save_file=False):
        model.eval()

        # get the prediction matrix
        pred_matrix = model.predict(data_normalization=self.data_norm, clip=True)
        pred_matrix = pred_matrix.detach().cpu().numpy()

        # get the test matrix
        test_ndmat = test_matrix.toarray()

        # filter out the unpredictable test ratings
        record_matrix = np.where(test_ndmat, 1, 0)  # only consider the purchased item to test the preference predict
        pred_matrix = np.multiply(record_matrix, pred_matrix)

        # save file
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
