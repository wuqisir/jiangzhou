import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import scipy
from utils.Others import regularization, revert_MinMaxScaler
from evaluate import recall_precision_f1_hr


class Interactions(data.Dataset):
    def __init__(self, ratmat, senmat, wvalmat):
        self.ratmat = ratmat.astype(np.float32).tocoo()
        self.senmat = senmat.astype(np.float32).tocoo()
        self.wvalmat = wvalmat.astype(np.float32).tocoo()
        self.n_users = self.ratmat.shape[0]
        self.n_items = self.ratmat.shape[1]

    def __getitem__(self, index):
        row = self.ratmat.row[index]  # different from getrow(index)
        col = self.ratmat.col[index]
        ratval = self.ratmat.data[index]
        senval = self.senmat.data[index]
        wval = self.wvalmat.data[index]
        return (row, col), ratval, senval, wval

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
    P, Q, = [], []
    for u in range(n_users):
        P.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    for i in range(n_items):
        Q.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    P = np.array(P)
    Q = np.array(Q)
    return P, Q,


class BaseModule(nn.Module):
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

    def loss(self, data):
        (rows, cols), ratval, senval, wval = data
        wval = wval - 1e-4  # recover the weight matrix

        # derive the rating-sens embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)

        # derive the regularization
        ues_reg = regularization(ues, self.reg_user)  # calculate the regularization of P
        uis_rat_reg = regularization(uis_rat, self.reg_item_rat)  # calculate the regularization of Q

        # preds for rating-sentiment
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)

        # sigmoid
        preds_rat_siged = F.sigmoid(preds_rat)
        # preds_rat_siged = preds_rat

        # calculate rat-sens loss
        # temp1 = wval
        # temp2 = (1 - wval)
        # temp3 = (preds_rat_siged.view(-1) - senval).pow(2)
        # temp4 = (preds_rat_siged.view(-1) - ratval).pow(2)
        # loss_rat = ((preds_rat_siged.view(-1) - ratval).pow(2) * (1 - wval)).sum()
        # loss_sen = ((preds_rat_siged.view(-1) - senval).pow(2) * wval * 0.2).sum()
        loss_rat = ((preds_rat_siged.view(-1) - ratval).pow(2) * wval).sum()
        loss_sen = ((preds_rat_siged.view(-1) - senval).pow(2) * (1 - wval)).sum()
        # loss_rat = ((preds_rat_siged.view(-1) - ratval).pow(2)).sum()
        # loss_rat = ((preds_rat_siged.squeeze() - ratval).pow(2)).sum()
        # loss_sen = ((preds_rat_siged.view(-1) - senval).pow(2)).sum()
        # loss_sen = 0
        loss = loss_sen + loss_rat + ues_reg + uis_rat_reg
        # if torch.isnan(loss):
        #     print('error')
        return loss

    def loss_rat_(self, data, data_normalization=False, datamin=1, datamax=5, clip=False):
        (rows, cols), ratval = data
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)
        preds_rat_siged = F.sigmoid(preds_rat)
        # preds_rat_siged = preds_rat
        if data_normalization:
            preds_rat_siged = revert_MinMaxScaler(preds_rat_siged, datamin, datamax)
        if clip:
            preds_rat_siged = torch.clamp(preds_rat_siged, datamin, datamax)
        loss_rat = (preds_rat_siged.squeeze() - ratval).pow(2).sum()
        return loss_rat

    def predict(self, data_normalization=False, datamin=1, datamax=5, clip=False):
        P = self.user_embedding.weight.data
        Q = self.item_rat_embedding.weight.data
        pred_matrix = torch.mm(P, Q.t())
        pred_matrix_siged = F.sigmoid(pred_matrix)
        # pred_matrix_siged = pred_matrix
        if data_normalization:
            pred_matrix_siged = revert_MinMaxScaler(pred_matrix_siged, datamin, datamax)
        if clip:
            pred_matrix_siged = torch.clamp(pred_matrix_siged, datamin, datamax)
        return pred_matrix_siged


class BasePipeline:
    def __init__(self,
                 train_rat_mat,
                 test_rat_mat,
                 sen_mat,
                 weight_mat,
                 model,
                 n_factors,
                 batch_size,
                 lr,
                 train_u_avg_dict,
                 test_u_items_dict,
                 sparse=False,
                 optimizer=torch.optim.Adam,
                 loss_function=nn.MSELoss(reduce=True, size_average=False),
                 n_epochs=10,
                 topk=5,
                 train_interaction_class=Interactions,
                 test_interaction_class=Interactions_test,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 data_normalization=False,
                 init_type='guassian'):
        self.train_rat = train_rat_mat
        self.test_rat = test_rat_mat
        self.sen_mat = sen_mat
        self.w_mat = weight_mat
        self.test_u_items_dict = test_u_items_dict
        self.train_u_avg_dict = train_u_avg_dict
        # define the train_data
        self.train_loader = data.DataLoader(train_interaction_class(train_rat_mat, sen_mat, weight_mat),
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

        # set the initial latent matrix
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
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.topk = topk
        self.data_norm = data_normalization
        self.results_recording = []
        self.rmse_results_recording = []

    # save results
    def saverecording(self, record, file, columns):
        pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)

    def fit(self, save_file=None, save_file_rmse=None):
        for epoch in range(1, self.n_epochs + 1):
            train_loss = self._fit_epoch(epoch)
            row = 'Epoch: {0:^3}  train:{1:^10.5f}'.format(epoch, train_loss)
            if self.test_rat is not None:
                test_rmse = self._validation_loss()
                row += 'val: {0:^10.5f}'.format(test_rmse)
                self.rmse_results_recording.append([epoch, test_rmse.item()])
                if epoch % 10 == 0:
                    top_k, recall, precision, f1_score, hr, ndcg = self._eval_metrics(self.model, self.test_rat,
                                                                                      self.train_u_avg_dict,
                                                                                      topk=self.topk)
                    print(
                        'e:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{}'.format(epoch, top_k,
                                                                                                      recall,
                                                                                                      precision,
                                                                                                      f1_score,
                                                                                                      hr, ndcg))
                    self.results_recording.append([epoch, recall, precision, f1_score, hr, ndcg])
            print(row)
        if save_file:
            self.saverecording(self.results_recording, save_file,
                               columns=['epoch', 'rec@{}'.format(self.topk), 'pre@{}'.format(self.topk),
                                        'f1@{}'.format(self.topk), 'hr', 'ndcg@{}'.format(self.topk)])
        if save_file_rmse:
            self.saverecording(self.rmse_results_recording, save_file_rmse,
                               columns=['epoch', 'RMSE'])
        return self.n_factors, test_rmse.item(), recall, precision, f1_score, hr, ndcg

    def _fit_epoch(self, epoch=1):
        self.model.train()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), ratval, senval, wval) in enumerate(self.train_loader):
            self.optimizer.zero_grad()  # set the gradient to zero

            row = row.long()  # long has the same meaning of int
            col = col.long()
            ratval = ratval.float()
            senval = senval.float()
            wval = wval.float()
            # preds = self.model(row, col)
            # loss = self.loss_function(preds, val)
            loss = self.model.loss(((row, col), ratval, senval, wval))
            loss.backward()

            # check the gradient
            # for param in self.model.named_parameters():
            #     if 'user' in param[0]:
            #         print(param[0], '---', param[1].grad[row], param[1].grad[row+1])
            # print('=='*12)

            self.optimizer.step()
            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
            # pbar.set_postfix(train_loss=batch_loss)
        total_loss /= self.train_rat.nnz
        return total_loss[0]

    def _validation_loss(self):
        self.model.eval()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            col = col.long()
            val = val.float()
            loss = self.model.loss_rat_(((row, col), val), data_normalization=self.data_norm, clip=True)
            total_loss += loss.item()
        total_loss /= self.test_rat.nnz
        rmse = torch.sqrt(total_loss[0])
        return rmse

    def _eval_metrics(self, model, test_matrix, train_u_avg, topk=5):
        model.eval()

        # get the prediction matrix
        pred_matrix = model.predict(data_normalization=self.data_norm, clip=True)
        pred_matrix = pred_matrix.detach().numpy()

        # get the test matrix
        test_ndmat = test_matrix.toarray()

        # filter out the unpredictable test ratings
        record_matrix = np.where(test_ndmat, 1, 0)  # only consider the purchased item to test the preference predict
        pred_matrix = np.multiply(record_matrix, pred_matrix)

        # evaluating
        top_k, recall, precision, f1_score, hr, ndcg = recall_precision_f1_hr(pred_matrix, test_ndmat,
                                                                              train_u_avg,
                                                                              top_k=topk)

        return top_k, recall, precision, f1_score, hr, ndcg
