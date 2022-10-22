import torch
from time import time
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scipy
import math
from tqdm import tqdm
from utils.Others import regularization, revert_MinMaxScaler
from evaluate import recall_precision_f1_hr


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
    # for u in range(n_users):
    #     P.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    # for i in range(n_items):
    #     Q.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    P = np.random.randn(n_users, n_factors) / np.sqrt(n_factors)
    Q = np.random.randn(n_items, n_factors) / np.sqrt(n_factors)
    # P = np.array(P) / 4
    # Q = np.array(Q) / 4
    return P, Q


def initPQAdjustedXavier(n_users, n_items, n_factors):
    """Let the sigmoid(PuTQi) is subject to Normal distribution with std=1"""
    P = np.random.randn(n_users, n_factors) * np.sqrt(2) / np.sqrt(n_factors)
    Q = np.random.randn(n_items, n_factors) * np.sqrt(2) / np.sqrt(n_factors)
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

    def loss(self, data):
        (rows, cols), ratval = data

        ratval = torch.where(ratval == -1, torch.Tensor([0]), ratval)  # convert the negative label -1 to 0

        positive_index = torch.nonzero(ratval).view(-1)
        negative_index = torch.nonzero(ratval - 1).view(-1)  # -1 is like ravel

        # derive the rating embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)

        # derive the positive and negative rating embeddings
        row_pos_index, row_neg_index = rows[positive_index], rows[negative_index]
        col_pos_index, col_neg_index = cols[positive_index], cols[negative_index]
        ues_rat_pos = self.user_embedding(row_pos_index)
        ues_rat_neg = self.user_embedding(row_neg_index)
        uis_rat_pos = self.item_rat_embedding(col_pos_index)
        uis_rat_neg = self.item_rat_embedding(col_neg_index)

        # derive the regularization
        ues_reg = regularization(ues, self.reg_user)  # calculate the regularization of P
        uis_rat_reg = regularization(uis_rat, self.reg_item_rat)  # calculate the regularization of Q

        # preds for reliability
        # TODO: simplize the following lines
        preds_rat_pos = (ues_rat_pos * uis_rat_pos).sum(dim=1, keepdim=True) if min(
            positive_index.shape) > 0 else torch.Tensor([0])
        preds_rat_neg = (ues_rat_neg * uis_rat_neg).sum(dim=1, keepdim=True) if min(
            negative_index.shape) > 0 else torch.Tensor([0])

        # sigmoid
        preds_rat_pos_siged = F.sigmoid(preds_rat_pos) if min(positive_index.shape) > 0 else torch.Tensor([0])
        preds_rat_neg_siged = F.sigmoid(preds_rat_neg) if min(negative_index.shape) > 0 else torch.Tensor([0])
        # calculate rat loss
        loss_rat_pos = torch.log(preds_rat_pos_siged).sum() if min(positive_index.shape) > 0 else torch.Tensor([0])
        loss_rat_neg = torch.log(1 - preds_rat_neg_siged).sum() if min(negative_index.shape) > 0 else torch.Tensor([0])
        loss = -1 * (loss_rat_pos + loss_rat_neg) + ues_reg + uis_rat_reg

        # if torch.isnan(loss):
        # print('error')
        return loss

    def predict(self):
        P = self.user_embedding.weight.data
        Q = self.item_rat_embedding.weight.data
        pred_matrix = torch.mm(P, Q.t())
        pred_matrix_siged = F.sigmoid(pred_matrix)
        return pred_matrix_siged

    def predict_instance(self, data):
        (rows, cols), ratval = data

        # derive the rating embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)

        # preds for rating
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)
        preds_rat_siged = F.sigmoid(preds_rat)

        return preds_rat_siged


class BasePipeline:
    def __init__(self,
                 train_rat_mat_iterator,
                 test_rat_mat,
                 n_factors,
                 batch_size,
                 lr,
                 train_u_avg_dict,
                 rate_range,
                 sparse=False,
                 n_epochs=10,
                 topk=5,
                 train_interaction_class=Interactions,
                 test_interaction_class=Interactions_test,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 data_normalization=False):
        self.train_rat = train_rat_mat_iterator
        self.test_rat = test_rat_mat
        self.train_u_avg_dict = train_u_avg_dict
        self.train_loader_dict = {}

        # initialize all the loc trainloaders
        for rate in rate_range:
            loc_train_mat = train_rat_mat_iterator[rate]
            loc_train_loader = data.DataLoader(train_interaction_class(loc_train_mat),
                                               batch_size=batch_size,
                                               shuffle=True)
            self.train_loader_dict[rate] = loc_train_loader
        self.test_loader = data.DataLoader(test_interaction_class(test_rat_mat), batch_size=batch_size,
                                           shuffle=True)
        # define some basic params
        self.n_users = self.train_rat[rate_range[0]].shape[0]
        self.n_items = self.train_rat[rate_range[0]].shape[1]
        self.n_factors = n_factors
        self.reg_user = reg_user
        self.reg_item_rat = reg_item_rat

        # set the initial latent matrix
        self.initP_dict = {}
        self.initQ_dict = {}
        for rate in rate_range:
            np.random.seed(3)
            # initP, initQ = initPQ(self.n_users, self.n_items, self.n_factors)
            initP, initQ = initPQAdjustedXavier(self.n_users, self.n_items, self.n_factors)
            # initP = np.load('./data/backup/initP.npy')
            # initQ = np.load('./data/backup/initQ.npy')
            initP = torch.from_numpy(initP)
            initQ = torch.from_numpy(initQ)
            initP = initP.float()
            initQ = initQ.float()
            self.initP_dict.setdefault(rate, initP)
            self.initQ_dict.setdefault(rate, initQ)

        # define the optimization params
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

        # initializing all loc models of certain rate.
        self.rate_range = rate_range
        self.model_dict = {}
        self.optimizer_dict = {}
        for rate in rate_range:
            initP = self.initP_dict[rate]
            initQ = self.initQ_dict[rate]
            loc_model = BaseModule(self.n_users,
                                   self.n_items,
                                   initP,
                                   initQ,
                                   n_factors=self.n_factors,
                                   sparse=sparse,
                                   reg_user=self.reg_user,
                                   reg_item_rat=self.reg_item_rat)
            loc_optimizer = torch.optim.SGD(loc_model.parameters(), lr=self.lr)
            self.model_dict.setdefault(rate, loc_model)
            self.optimizer_dict.setdefault(rate, loc_optimizer)

        self.topk = topk
        self.data_norm = data_normalization
        self.results_recording = []
        self.rmse_results_recording = []

    # save results
    def saverecording(self, record, file, columns):
        pd.DataFrame(record, columns=columns).to_csv(file, sep=',', header=True, index=False)

    def fit(self, save_file=None, save_file_rmse=None, rank_metric=False):
        for epoch in range(1, self.n_epochs + 1):
            # row = 'Epoch: {0:^3}  train:{1:^10.5f}'.format(epoch, train_loss)
            # print(row)
            row = ''
            pred_mat = None
            for rate in self.model_dict.keys():
                # print('training rate={} model'.format(rate))
                # print('=' * 20)
                s_time = time()
                trainloss = self._fit_epoch(rate)
                print(f"training_time: {time() - s_time}")
                # print('training loss:{}'.format(trainloss))

            # self.model_dict[1].eval()
            # self.model_dict[2].eval()
            # print(id(self.model_dict[1]))
            # print(id(self.model_dict[2]))
            # print(id(self.model_dict[3]))
            # print(id(self.model_dict[4]))
            # print(id(self.model_dict[5]))
            # temp1 = self.model_dict[1].user_embedding.weight.data
            # temp2 = self.model_dict[2].user_embedding.weight.data
            # temp3 = self.model_dict[3].user_embedding.weight.data
            # temp4 = self.model_dict[4].user_embedding.weight.data
            # temp5 = self.model_dict[5].user_embedding.weight.data
            # r12 = (temp1 == temp2).bool()
            # r13 = (temp1 == temp3).bool()
            # r14 = (temp1 == temp4).bool()
            # r15 = (temp1 == temp5).bool()
            # if torch.all(r12):
            #     print('model1 and model2 are totoally equal')
            # if torch.all(r13):
            #     print('model1 and model3 are totoally equal')
            # if torch.all(r14):
            #     print('model1 and model4 are totoally equal')
            # if torch.all(r15):
            #     print('model1 and model5 are totoally equal')
            # temp1 = self.model_dict[1].item_rat_embedding.weight.data
            # temp2 = self.model_dict[2].item_rat_embedding.weight.data
            # temp3 = self.model_dict[3].item_rat_embedding.weight.data
            # temp4 = self.model_dict[4].item_rat_embedding.weight.data
            # temp5 = self.model_dict[5].item_rat_embedding.weight.data
            # r12 = (temp1 == temp2).bool()
            # r13 = (temp1 == temp3).bool()
            # r14 = (temp1 == temp4).bool()
            # r15 = (temp1 == temp5).bool()
            # if torch.all(r12):
            #     print('item model1 and model2 are totoally equal')
            # if torch.all(r13):
            #     print('item model1 and model3 are totoally equal')
            # if torch.all(r14):
            #     print('item model1 and model4 are totoally equal')
            # if torch.all(r15):
            #     print('item model1 and model5 are totoally equal')

            if self.test_rat is not None:
                pred_mat, test_rmse = self._validation_loss()
                row += 'epoch: {0} val: {1:^10.5f}'.format(epoch, test_rmse)
                self.rmse_results_recording.append([epoch, test_rmse.item()])
            if epoch % 10 == 0:
                print(row)
            if rank_metric and (epoch % 10 == 0):
                top_k, recall, precision, f1_score, hr, ndcg = self._eval_metrics(pred_mat, self.test_rat,
                                                                                  self.train_u_avg_dict,
                                                                                  topk=self.topk)
                print(
                    'e:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{}'.format(epoch, top_k,
                                                                                                  recall,
                                                                                                  precision,
                                                                                                  f1_score,
                                                                                                  hr, ndcg))
                self.results_recording.append([epoch, recall, precision, f1_score, hr, ndcg])
        if save_file:
            self.saverecording(self.results_recording, save_file,
                               columns=['epoch', 'rec@{}'.format(self.topk), 'pre@{}'.format(self.topk),
                                        'f1@{}'.format(self.topk), 'hr', 'ndcg@{}'.format(self.topk)])
        if save_file_rmse:
            self.saverecording(self.rmse_results_recording, save_file_rmse,
                               columns=['epoch', 'RMSE'])

        return self.n_factors, test_rmse.item(), recall, precision, f1_score, hr, ndcg

    def _fit_epoch(self, rate):
        model = self.model_dict[rate]
        train_loader = self.train_loader_dict[rate]
        optimizer = self.optimizer_dict[rate]
        model.train()
        total_loss = torch.Tensor([0])
        for batch_idx, ((row, col), ratval) in enumerate(train_loader):
            optimizer.zero_grad()  # set the gradient to zero

            row = row.long()  # long has the same meaning of int
            col = col.long()
            ratval = ratval.float()

            loss = model.loss(((row, col), ratval))
            loss.backward()

            # print('model:{}'.format(rate))
            # print('before update')
            # print('=' * 80)
            # for param in model.named_parameters():
            #     if 'user' in param[0]:
            #         print(param[0], '---', param[1].grad[row], param[1][row])
            # print('==' * 12)
            #
            # temp1 = self.model_dict[1].user_embedding.weight.data[row]
            # temp2 = self.model_dict[2].user_embedding.weight.data[row]
            # temp3 = self.model_dict[3].user_embedding.weight.data[row]
            # temp4 = self.model_dict[4].user_embedding.weight.data[row]
            # temp5 = self.model_dict[5].user_embedding.weight.data[row]
            # print('model1:{}'.format(temp1))
            # print('model2:{}'.format(temp2))
            # print('model3:{}'.format(temp3))
            # print('model4:{}'.format(temp4))
            # print('model5:{}'.format(temp5))
            #
            # print('embedding weight data id:')
            # print(id(self.model_dict[1].user_embedding.weight.data))
            # print(id(self.model_dict[2].user_embedding.weight.data))
            # print(id(self.model_dict[3].user_embedding.weight.data))
            # print(id(self.model_dict[4].user_embedding.weight.data))
            # print(id(self.model_dict[5].user_embedding.weight.data))
            #
            # print('embedding id:')
            # print(id(self.model_dict[1].user_embedding))
            # print(id(self.model_dict[2].user_embedding))
            # print(id(self.model_dict[3].user_embedding))
            # print(id(self.model_dict[4].user_embedding))
            # print(id(self.model_dict[5].user_embedding))
            #
            # print('optimizer id')
            # print(id(self.optimizer_dict[1]))
            # print(id(self.optimizer_dict[2]))
            # print(id(self.optimizer_dict[3]))
            # print(id(self.optimizer_dict[4]))
            # print(id(self.optimizer_dict[5]))

            optimizer.step()  # upgrade the gradient

            # check the gradient
            # print('after update')
            # print('=' * 80)
            # for param in model.named_parameters():
            #     if 'user' in param[0]:
            #         print(param[0], '---', param[1].grad[row], param[1][row])
            # print('==' * 12)
            #
            # temp1 = self.model_dict[1].user_embedding.weight.data[row]
            # temp2 = self.model_dict[2].user_embedding.weight.data[row]
            # temp3 = self.model_dict[3].user_embedding.weight.data[row]
            # temp4 = self.model_dict[4].user_embedding.weight.data[row]
            # temp5 = self.model_dict[5].user_embedding.weight.data[row]
            # print('model1:{}'.format(temp1))
            # print('model2:{}'.format(temp2))
            # print('model3:{}'.format(temp3))
            # print('model4:{}'.format(temp4))
            # print('model5:{}'.format(temp5))
            # print(id(self.model_dict[1]))
            # print(id(self.model_dict[2]))
            # print(id(self.model_dict[3]))
            # print(id(self.model_dict[4]))
            # print(id(self.model_dict[5]))

            total_loss += loss.item()
            batch_loss = loss.item() / row.size()[0]
        # TODO: change the number of the ratings
        total_loss /= self.train_rat[self.rate_range[0]].nnz
        return total_loss

    def _validation_loss(self):
        total_loss = torch.Tensor([0])
        total_preds = []
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long()
            col = col.long()
            val = val.float()
            preds_rat, loss = self._total_loss_rat(((row, col), val))
            total_loss += loss.item()
            total_preds.extend(preds_rat)  # collect the predictions
        total_preds = np.array(total_preds)
        total_loss /= self.test_rat.nnz
        rmse = torch.sqrt(total_loss[0])

        # get the coo_matrix of the predictions
        row, col, val = total_preds[:, 0], total_preds[:, 1], total_preds[:, 2]
        total_pred_mat = sp.coo_matrix((val, (row, col)), dtype=np.int)
        total_pred_mat = total_pred_mat.toarray()

        return total_pred_mat, rmse  # return the whole predictions and rmse

    def _total_loss_rat(self, data):
        (row, col), ratval = data
        loc_pred_list = []

        # each model make the predictions on the same point
        for rate in self.rate_range:
            model = self.model_dict[rate]
            preds_rat_siged = model.predict_instance(data)
            temp = preds_rat_siged.view(-1)
            temp1 = temp.detach().numpy()
            loc_pred_list.append(preds_rat_siged.view(-1).detach().numpy())

        pred_rel = np.array(loc_pred_list)
        indexes = np.argmax(pred_rel, axis=0)  # which model make the most reliable prediction
        preds_rat_arr = self.rate_range[indexes]
        preds_rat = torch.from_numpy(preds_rat_arr).unsqueeze(dim=-1)  # back to tensor
        ratval = ratval.int()
        loss_rat = (preds_rat.squeeze() - ratval).pow(2).sum()
        predictions = np.vstack([row, col, preds_rat_arr]).T
        return predictions, loss_rat

    def _eval_metrics(self, pred_matrix, test_matrix, train_u_avg, topk=5):

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
