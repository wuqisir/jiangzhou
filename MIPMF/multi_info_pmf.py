import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
from time import time
import scipy
from utils.Others import regularization, revert_MinMaxScaler
from evaluate import recall_precision_f1_hr

UNREL_USERS = [4501, 4470, 3794, 1765, 4134, 3400, 674, 2095, 716, 4937, 3221,
               1696, 4246, 2203]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MT = True
PRED_SAVEFILE = './pred_array_MIPMF'
TEST_SAVEFILE = './test_array_MIPMF'


def lookup_unrel(rel_mat):
    zero_index = np.where(rel_mat.data == 1e-4)
    users = rel_mat.row[zero_index]
    temp = pd.Series(users).value_counts()
    users = temp[temp >= 20].index
    print(temp)
    print(users)
    return users


class Interactions(data.Dataset):
    def __init__(self, ratmat, relmat, senmat, senhelmat):
        self.ratmat = ratmat.astype(np.float32).tocoo()
        self.relmat = relmat.astype(np.float32).tocoo()  # rel should be float since rel is added 1e-4
        # lookup_unrel(self.relmat)
        # exit()
        self.senmat = senmat.astype(np.float32).tocoo()
        self.senhelmat = senhelmat.astype(np.float32).tocoo()
        self.n_users = self.ratmat.shape[0]
        self.n_items = self.ratmat.shape[1]

    def __getitem__(self, index):
        row = self.ratmat.row[index]  # different from getrow(index)
        col = self.ratmat.col[index]
        ratval = self.ratmat.data[index]
        relval = self.relmat.data[index]
        senval = self.senmat.data[index]
        senhelval = self.senhelmat.data[index]
        return (row, col), ratval, relval, senval, senhelval

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


def initPQZ(n_users, n_items, n_factors):
    P, Q, Z = [], [], []
    for u in range(n_users):
        P.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    for i in range(n_items):
        Q.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    for i in range(n_items):
        Z.append(np.array([np.random.random() / math.sqrt(n) for n in range(1, n_factors + 1)]))
    P = np.array(P)
    Q = np.array(Q)
    Z = np.array(Z)
    return P, Q, Z


def initPQZ_Guassian(n_users, n_items, n_factors):
    P = 0.1 * np.random.randn(n_users, n_factors)
    Q = 0.1 * np.random.randn(n_items, n_factors)
    Z = 0.1 * np.random.randn(n_items, n_factors)
    return P, Q, Z


def get_weight(sen_hel_val, rel_val, eps, ws):
    """Compute the element-wise weight of sentiment and rating
    Args:
        sen_hel_val: 1D tensor
        rel_val: 1D tensor
        eps: helpness threshold
        ws: weight of sentiment
        return 1D tensors: sentiment_weight & rating_weight
    """
    # compute the flag indicating different cases
    # An example like below:
    # sen_hel_val = torch.Tensor([0.9, 0.1, 0.9, 0.1])
    # rel_val = torch.Tensor([1., 1., 0., 0.])
    sen_rel_val = torch.where(sen_hel_val >= eps, torch.Tensor([1]).to(DEVICE),
                              torch.Tensor([0]).to(DEVICE))  # rel_val is float since it may compute with float numbers
    flag = sen_rel_val + rel_val
    rel_rel_flag = torch.where(flag == 2., torch.Tensor([1]).to(DEVICE), torch.Tensor([0]).to(DEVICE))
    unrel_unrel_flag = torch.where(flag == 0., torch.Tensor([1]).to(DEVICE), torch.Tensor([0]).to(DEVICE))

    # compute sentiment weight
    sen_rel_rel_weight = torch.mul((ws - 1) * torch.ones(sen_rel_val.shape[0]).to(DEVICE), rel_rel_flag)
    sen_unrel_unrel_weight = torch.mul(0. * torch.ones(sen_rel_val.shape[0]).to(DEVICE), unrel_unrel_flag)
    sen_weight = sen_rel_val + sen_rel_rel_weight + sen_unrel_unrel_weight

    # compute rating weight
    rat_rel_rel_weight = torch.mul((-ws) * torch.ones(sen_rel_val.shape[0]).to(DEVICE), rel_rel_flag)
    rat_unrel_unrel_weight = sen_unrel_unrel_weight
    rat_weight = rel_val + rat_rel_rel_weight + rat_unrel_unrel_weight

    return sen_weight, rat_weight


class BaseModule(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 initP,
                 initQ,
                 initZ,
                 n_factors=40,
                 weight_sen=0.7,
                 weight_rel=0.3,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 reg_item_rel=0.001,
                 hel_eps=None,
                 sparse=False):
        super(BaseModule, self).__init__()  # inherit the vars and functions in the super class
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_embedding = nn.Embedding(self.n_users, self.n_factors, _weight=initP, sparse=sparse)
        self.item_rat_embedding = nn.Embedding(self.n_items, self.n_factors, _weight=initQ, sparse=sparse)
        self.item_rel_embedding = nn.Embedding(self.n_items, self.n_factors, _weight=initZ, sparse=sparse)
        self.sparse = sparse
        self.ws = weight_sen
        self.wr = weight_rel
        self.heleps = hel_eps
        self.reg_user = reg_user
        self.reg_item_rat = reg_item_rat
        self.reg_item_rel = reg_item_rel

    def loss(self, data):
        (rows, cols), ratval, relval, senval, senhelval = data
        # vals = torch.where(vals == -1, torch.Tensor([0]), vals)  # convert the negative label -1 to 0
        relval = relval - 1e-4  # recover the binary matrix
        senhelval = senhelval - 1e-4  # recover the helpness matrix containing zero
        positive_index = torch.nonzero(relval).view(-1)
        negative_index = torch.nonzero(relval - 1).view(-1)  # view(-1) is like ravel

        # derive the reliability embeddings
        row_pos_index, row_neg_index = rows[positive_index], rows[negative_index]
        col_pos_index, col_neg_index = cols[positive_index], cols[negative_index]
        ues_rel_pos = self.user_embedding(row_pos_index)
        ues_rel_neg = self.user_embedding(row_neg_index)
        uis_rel_pos = self.item_rel_embedding(col_pos_index)
        uis_rel_neg = self.item_rel_embedding(col_neg_index)

        # derive the rating-sens embeddings
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)
        uis_rel = self.item_rel_embedding(cols)

        # derive the regularization
        ues_reg = regularization(ues, self.reg_user)  # calculate the regularization of P
        uis_rat_reg = regularization(uis_rat, self.reg_item_rat)  # calculate the regularization of Q
        uis_rel_reg = regularization(uis_rel, self.reg_item_rel)  # calculate the regularization of Z

        # preds for rating-sentiment
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)

        # preds for reliability
        # TODO: simplize the following lines
        preds_rel_pos = (ues_rel_pos * uis_rel_pos).sum(dim=1, keepdim=True) if min(
            positive_index.shape) > 0 else torch.Tensor([0]).to(DEVICE)
        preds_rel_neg = (ues_rel_neg * uis_rel_neg).sum(dim=1, keepdim=True) if min(
            negative_index.shape) > 0 else torch.Tensor([0]).to(DEVICE)

        # sigmoid
        preds_rat_siged = F.sigmoid(preds_rat)
        preds_rel_pos_siged = F.sigmoid(preds_rel_pos) if min(positive_index.shape) > 0 else torch.Tensor([0]).to(
            DEVICE)
        preds_rel_neg_siged = F.sigmoid(preds_rel_neg) if min(negative_index.shape) > 0 else torch.Tensor([0]).to(
            DEVICE)

        # calculate rel loss
        loss_rel_pos = torch.log(preds_rel_pos_siged).sum() if min(positive_index.shape) > 0 else torch.Tensor([0]).to(
            DEVICE)
        loss_rel_neg = torch.log(1 - preds_rel_neg_siged).sum() if min(negative_index.shape) > 0 else torch.Tensor(
            [0]).to(DEVICE)
        loss_rel = loss_rel_pos + loss_rel_neg

        # get weights for rating and sentiment
        sen_weight, rat_weight = get_weight(senhelval, relval, self.heleps, self.ws)

        # calculate rat-sens loss
        # loss_rat = ((preds_rat_siged.view(-1) - ratval).pow(2)).sum()
        # loss_sen = ((preds_rat_siged.view(-1) - senval).pow(2) * senhelval).sum()
        # loss = self.ws * loss_sen + (1 - self.ws) * loss_rat - \
        #        self.wr * loss_rel + \
        #        ues_reg + uis_rat_reg + uis_rel_reg

        # calculate rat-sens loss with element-wise weights
        loss_rat = ((preds_rat_siged.view(-1) - ratval).pow(2) * rat_weight).sum()
        loss_sen = ((preds_rat_siged.view(-1) - senval).pow(2) * sen_weight).sum()
        loss = loss_sen + loss_rat - self.wr * loss_rel + ues_reg + uis_rat_reg + uis_rel_reg
        if torch.isnan(loss):
            print('error')
        return loss

    def loss_rat_(self, data, data_normalization=False, datamin=1, datamax=5, clip=False):
        (rows, cols), ratval = data
        ues = self.user_embedding(rows)
        uis_rat = self.item_rat_embedding(cols)
        preds_rat = (ues * uis_rat).sum(dim=1, keepdim=True)
        preds_rat_siged = F.sigmoid(preds_rat)
        if data_normalization:
            preds_rat_siged = revert_MinMaxScaler(preds_rat_siged, datamin, datamax)
        if clip:
            preds_rat_siged = torch.clamp(preds_rat_siged, datamin, datamax)
        loss_rat = (preds_rat_siged.squeeze() - ratval).pow(2).sum()
        return loss_rat

    # def predict(self, data_normalization=False, datamin=1, datamax=5, clip=False):
    #     P = self.user_embedding.weight.data
    #     Q = self.item_rat_embedding.weight.data
    #     pred_matrix = torch.mm(P, Q.t())
    #     pred_matrix_siged = F.sigmoid(pred_matrix)
    #     if data_normalization:
    #         pred_matrix_siged = revert_MinMaxScaler(pred_matrix_siged, datamin, datamax)
    #     if clip:
    #         pred_matrix_siged = torch.clamp(pred_matrix_siged, datamin, datamax)
    #     return pred_matrix_siged

    def predict(self, data_normalization=False, datamin=1, datamax=5, clip=False):
        P = self.user_embedding.weight.data
        Q = self.item_rat_embedding.weight.data
        if IS_MT:
            P = P.detach().cpu().numpy()
            Q = Q.detach().cpu().numpy()
            pred_matrix = np.matmul(P, Q.T)
            pred_matrix_siged = 1 / (1 + np.exp(-pred_matrix))
            if data_normalization:
                pred_matrix_siged = revert_MinMaxScaler(pred_matrix_siged, datamin, datamax)
            if clip:
                pred_matrix_siged = np.clip(pred_matrix_siged, datamin, datamax)
        else:
            pred_matrix = torch.mm(P, Q.t())
            pred_matrix_siged = F.sigmoid(pred_matrix)
            if data_normalization:
                pred_matrix_siged = revert_MinMaxScaler(pred_matrix_siged, datamin, datamax)
            if clip:
                pred_matrix_siged = torch.clamp(pred_matrix_siged, datamin, datamax)
        return pred_matrix_siged

    def predict_rel(self, theta):
        P = self.user_embedding.weight.data
        B = self.item_rel_embedding.weight.data
        if IS_MT:
            P = P.detach().cpu().numpy()
            B = B.detach().cpu().numpy()
            pred_rel_matrix = np.matmul(P, B.T)
            pred_rel_matrix_siged = 1 / (1 + np.exp(-pred_rel_matrix))
            pred_rel_matrix_binary = np.where(pred_rel_matrix_siged < theta, 0, 1)
        else:
            pred_rel_matrix = torch.mm(P, B.t())
            pred_rel_matrix_siged = F.sigmoid(pred_rel_matrix)
            pred_rel_matrix_binary = torch.where(pred_rel_matrix_siged < theta, torch.Tensor([0]).to(DEVICE),
                                                 torch.Tensor([1]).to(DEVICE))
        return pred_rel_matrix_siged, pred_rel_matrix_binary

    def predict_rel_instance(self, rows, cols, theta):
        ues = self.user_embedding(rows)
        uis_rel = self.item_rel_embedding(cols)
        pred_rel = (ues * uis_rel).sum(dim=1, keepdim=True)
        pred_rel_siged = F.sigmoid(pred_rel)
        pred_rel_binary = torch.where(pred_rel_siged < theta, torch.Tensor([0]).to(DEVICE),
                                      torch.Tensor([1]).to(DEVICE))
        return pred_rel_siged, pred_rel_binary

    # def forward(self, index):
    #     rows, cols = index
    #     ues = self.user_embedding(rows)
    #     uis = self.item_embedding(cols)
    #     preds = (ues * uis).sum(dim=1, keepdim=True)
    #     preds = preds.squeeze()
    #     return preds
    #

    #
    # def __call__(self, *args):
    #     return self.forward(*args)


class BasePipeline:
    def __init__(self,
                 train_rat_mat,
                 test_rat_mat,
                 rel_mat,
                 sen_mat,
                 sen_hel_mat,
                 model,
                 n_factors,
                 batch_size,
                 lr,
                 sparse=False,
                 optimizer=torch.optim.SGD,
                 loss_function=nn.MSELoss(reduce=True, size_average=False),
                 n_epochs=10,
                 topk=5,
                 train_interaction_class=Interactions,
                 test_interaction_class=Interactions_test,
                 weight_sentiment=0.7,
                 weight_reliability=0.3,
                 hel_eps=None,
                 reg_user=0.001,
                 reg_item_rat=0.001,
                 reg_item_rel=0.001,
                 data_normalization=False,
                 rel_theta=0.5,
                 seed=0):
        self.train_rat = train_rat_mat
        self.test_rat = test_rat_mat
        self.rel_mat = rel_mat
        self.sen_mat = sen_mat

        # define the train_data
        self.train_loader = data.DataLoader(train_interaction_class(train_rat_mat, rel_mat, sen_mat, sen_hel_mat),
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
        self.reg_item_rel = reg_item_rel
        # self.init = 'gaussian'
        self.init = 'xavier'
        self.seed = seed

        # set the initial latent matrix
        np.random.seed(self.seed)
        if self.init == 'gaussian':
            initP, initQ, initZ = initPQZ_Guassian(self.n_users, self.n_items, self.n_factors)
        else:
            initP, initQ, initZ = initPQZ(self.n_users, self.n_items, self.n_factors)

        # np.random.seed(0)
        # _, _, initZ = initPQZ(self.n_users, self.n_items, self.n_factors)
        # initP = np.load('./data/backup/initP.npy')
        # initQ = np.load('./data/backup/initQ.npy')
        initP = torch.from_numpy(initP)
        initQ = torch.from_numpy(initQ)
        initZ = torch.from_numpy(initZ)
        initP = initP.float()
        initQ = initQ.float()
        initZ = initZ.float()

        # define the optimization params
        self.batch_size = batch_size
        self.lr = lr
        self.loss_function = loss_function  # define the loss function type
        self.n_epochs = n_epochs
        self.ws = weight_sentiment
        self.wr = weight_reliability
        self.model = model(self.n_users,
                           self.n_items,
                           initP,
                           initQ,
                           initZ,
                           n_factors=self.n_factors,
                           sparse=sparse,
                           reg_user=self.reg_user,
                           reg_item_rat=self.reg_item_rat,
                           reg_item_rel=self.reg_item_rel,
                           weight_sen=weight_sentiment,
                           weight_rel=weight_reliability,
                           hel_eps=hel_eps)
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, momentum=0.97)
        # self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.topk = topk
        self.data_norm = data_normalization
        self.rel_theta = rel_theta
        self.results_recording = []
        self.rmse_results_recording = []
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(DEVICE)

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

                # Target RMSE
                target_rmse = 1.011
                if (target_rmse - 0.05) <= test_rmse <= (target_rmse + 0.05):
                    # get the prediction matrix
                    pred_matrix = self.model.predict(data_normalization=self.data_norm, clip=True)
                    if IS_MT:
                        # get the predict reliable matrix, reliable binary matrix
                        pred_matrix_rel, pred_matrix_rel_bi = self.model.predict_rel(self.rel_theta)
                    else:
                        pred_matrix = pred_matrix.detach().cpu().numpy()

                        # get the predict reliable matrix, reliable binary matrix
                        pred_matrix_rel, pred_matrix_rel_bi = self.model.predict_rel(self.rel_theta)
                        pred_matrix_rel = pred_matrix_rel.detach().cpu().numpy()
                        pred_matrix_rel_bi = pred_matrix_rel_bi.detach().cpu().numpy()

                    # filter out the unreliable prediction
                    pred_matrix = np.multiply(pred_matrix, pred_matrix_rel_bi)

                    # get the test matrix
                    # record_matrix = np.where(train_matrix, 0, 1)
                    test_ndmat = self.test_rat.toarray()
                    # print('test entries:{}'.format(len(test_ndmat.nonzero()[0])))
                    test_ndmat = np.multiply(pred_matrix_rel_bi, test_ndmat)
                    # print('filtered test entries:{}'.format(len(test_ndmat.nonzero()[0])))

                    # filter out the unpredictable test rating
                    record_matrix = np.where(test_ndmat, 1,
                                             0)  # only consider the purchased item to test the preference predict
                    pred_matrix = np.multiply(record_matrix, pred_matrix)

                    test_array = test_ndmat[test_ndmat.nonzero()]
                    pred_array = pred_matrix[pred_matrix.nonzero()]

                    # save file
                    # np.save(PRED_SAVEFILE + '_' + str(epoch), pred_matrix)
                    # np.save(TEST_SAVEFILE + '_' + str(epoch), test_ndmat)
                    # print('SAVE FILE SUCCESS!')

                    # np.save(PRED_SAVEFILE + '_' + str(epoch) + '_' + str(test_rmse.numpy()), pred_array)
                    # np.save(TEST_SAVEFILE + '_' + str(epoch) + '_' + str(test_rmse.numpy()), test_array)
                    np.save(PRED_SAVEFILE + '_' + str(epoch),pred_array)
                    np.save(TEST_SAVEFILE + '_' + str(epoch),test_array)
                    print('SAVE FILE SUCCESS!')

                if (epoch % 10000 == 0):
                    save_flag = True if epoch == self.n_epochs else False
                    top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = self._eval_metrics(self.model,
                                                                                                         self.train_rat,
                                                                                                         self.test_rat,
                                                                                                         topk=self.topk,
                                                                                                         save_file=save_flag)
                    print(
                        'e:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{}, ppr:{}, ks:{}, r_square:{}'.format(
                            epoch,
                            top_k,
                            recall,
                            precision,
                            f1_score,
                            hr, ndcg,
                            ppr, ks, r_square))
                    self.results_recording.append([epoch, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square])

                # spec_test_rat = self.test_rat.tocsr()[UNREL_USERS].tocoo()
                # spec_test_loader = data.DataLoader(Interactions_test(spec_test_rat), batch_size=self.batch_size,
                #                                    shuffle=True)
                # test_rmse = self._validation_loss_specific(spec_test_rat, spec_test_loader)
                # row += 'val_spec: {0:^10.5f}'.format(test_rmse)
                # if (epoch % 10 == 0):
                #     top_k, recall, precision, f1_score, hr, ndcg = self._eval_metrics_spec(self.model, self.train_rat,
                #                                                                            spec_test_rat,
                #                                                                            self.train_u_avg_dict,
                #                                                                            topk=self.topk)
                #     print(
                #         'spec: e:{}, top_k:{}, recall:{}, precision:{}, f1_score:{}, hr:{}, ndcg:{}'.format(epoch,
                #                                                                                             top_k,
                #                                                                                             recall,
                #                                                                                             precision,
                #                                                                                             f1_score,
                #                                                                                             hr, ndcg))
                # self.results_recording.append([epoch, recall, precision, f1_score, hr, ndcg])

            print(row)
        if save_file:
            self.saverecording(self.results_recording, save_file,
                               columns=['epoch', 'rec@{}'.format(self.topk), 'pre@{}'.format(self.topk),
                                        'f1@{}'.format(self.topk), 'hr@{}'.format(self.topk),
                                        'ndcg@{}'.format(self.topk), 'ppr', 'ks', 'r_square'])
        if save_file_rmse:
            self.saverecording(self.rmse_results_recording, save_file_rmse,
                               columns=['epoch', 'RMSE'])
        if len(self.results_recording) == 0:
            return self.n_factors
        else:
            return self.n_factors, test_rmse.item(), recall, precision, f1_score, hr, ndcg, ppr, ks, r_square

    def _fit_epoch(self, epoch=1):
        model = self.model
        model.train()
        total_loss = torch.Tensor([0]).to(DEVICE)
        for batch_idx, ((row, col), ratval, relval, senval, senhelval) in enumerate(self.train_loader):
            print('\r' + 'epoch:' + str(epoch) + str(batch_idx) + '/' + str(len(self.train_loader)), end='', flush=True)
            self.optimizer.zero_grad()  # set the gradient to zero

            row = row.long().to(DEVICE)  # long has the same meaning of int
            col = col.long().to(DEVICE)
            ratval = ratval.float().to(DEVICE)
            relval = relval.float().to(DEVICE)
            senval = senval.float().to(DEVICE)
            senhelval = senhelval.float().to(DEVICE)
            # preds = self.model(row, col)
            # loss = self.loss_function(preds, val)
            loss = self.model.loss(((row, col), ratval, relval, senval, senhelval))
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

    def _validation_loss_specific(self, spec_test_rat, spec_test_loader):
        self.model.eval()
        total_loss = torch.Tensor([0]).to(DEVICE)
        total_reliables, total_unreliables = 0, 0
        for batch_idx, ((row, col), val) in enumerate(spec_test_loader):
            row = row.long()
            col = col.long()
            val = val.float()
            pred_rel, pred_rel_bi = self.model.predict_rel_instance(row, col, self.rel_theta)
            rel_idx = pred_rel_bi.view(-1)
            unreliables = len(rel_idx[rel_idx == 0])
            reliables = len(rel_idx) - unreliables
            filtered_row = row[rel_idx > 0]
            filtered_col = col[rel_idx > 0]
            filtered_val = val[rel_idx > 0]
            loss = self.model.loss_rat_(((filtered_row, filtered_col), filtered_val),
                                        data_normalization=self.data_norm,
                                        clip=True)
            total_unreliables += unreliables
            total_reliables += reliables
            total_loss += loss.item()
        total_loss /= spec_test_rat.nnz
        rmse = torch.sqrt(total_loss[0])
        print('total_spec_reliable_predictions:{}'.format(total_reliables))
        print('total_spec_unreliable_predictions:{}'.format(total_unreliables))
        return rmse

    def _validation_loss(self):
        # TODO: get the prediction of the rel.
        self.model.eval()
        total_loss = torch.Tensor([0]).to(DEVICE)
        total_reliables, total_unreliables = 0, 0
        for batch_idx, ((row, col), val) in enumerate(self.test_loader):
            row = row.long().to(DEVICE)
            col = col.long().to(DEVICE)
            val = val.float().to(DEVICE)
            # preds = self.model(row, col)
            # loss = self.loss_function(preds, val)
            pred_rel, pred_rel_bi = self.model.predict_rel_instance(row, col, self.rel_theta)
            rel_idx = pred_rel_bi.view(-1)
            unreliables = len(rel_idx[rel_idx == 0])
            reliables = len(rel_idx) - unreliables
            filtered_row = row[rel_idx > 0]
            filtered_col = col[rel_idx > 0]
            filtered_val = val[rel_idx > 0]
            loss = self.model.loss_rat_(((filtered_row, filtered_col), filtered_val),
                                        data_normalization=self.data_norm,
                                        clip=True)
            total_unreliables += unreliables
            total_reliables += reliables
            total_loss += loss.item()
        # total_loss /= self.test_rat.nnz
        total_loss /= total_reliables
        rmse = torch.sqrt(total_loss[0])
        print('total_reliable_predictions:{}'.format(total_reliables))
        print('total_unreliable_predictions:{}'.format(total_unreliables))
        print('coverage:{}'.format(total_reliables / (total_reliables + total_unreliables)))
        return rmse

    def _eval_metrics_spec(self, model, train_sp_matrix, test_matrix, train_u_avg, topk=5):
        model.eval()
        train_matrix = train_sp_matrix.toarray()

        # get the prediction matrix
        pred_matrix = model.predict(data_normalization=self.data_norm, clip=True)
        pred_matrix = pred_matrix.detach().numpy()[UNREL_USERS]

        # get the predict reliable matrix, reliable binary matrix
        pred_matrix_rel, pred_matrix_rel_bi = model.predict_rel(self.rel_theta)
        pred_matrix_rel = pred_matrix_rel.detach().numpy()[UNREL_USERS]
        pred_matrix_rel_bi = pred_matrix_rel_bi.detach().numpy()[UNREL_USERS]

        # filter out the unreliable prediction
        pred_matrix = np.multiply(pred_matrix, pred_matrix_rel_bi)

        # get the test matrix
        # record_matrix = np.where(train_matrix, 0, 1)
        test_ndmat = test_matrix.toarray()
        print('test entries:{}'.format(len(test_ndmat.nonzero()[0])))
        test_ndmat = np.multiply(pred_matrix_rel_bi, test_ndmat)
        print('filtered test entries:{}'.format(len(test_ndmat.nonzero()[0])))

        # filter out the unpredictable test rating
        record_matrix = np.where(test_ndmat, 1, 0)  # only consider the purchased item to test the preference predict
        pred_matrix = np.multiply(record_matrix, pred_matrix)

        # evaluating
        top_k, recall, precision, f1_score, hr, ndcg, ppr = recall_precision_f1_hr(pred_matrix, pred_matrix_rel,
                                                                                   test_ndmat,
                                                                                   train_u_avg,
                                                                                   top_k=topk)

        return top_k, recall, precision, f1_score, hr, ndcg, ppr

    def _eval_metrics(self, model, train_sp_matrix, test_matrix, topk=5, save_file=True):
        model.eval()
        train_matrix = train_sp_matrix.toarray()

        # get the prediction matrix
        pred_matrix = model.predict(data_normalization=self.data_norm, clip=True)
        if IS_MT:
            # get the predict reliable matrix, reliable binary matrix
            pred_matrix_rel, pred_matrix_rel_bi = model.predict_rel(self.rel_theta)
        else:
            pred_matrix = pred_matrix.detach().cpu().numpy()

            # get the predict reliable matrix, reliable binary matrix
            pred_matrix_rel, pred_matrix_rel_bi = model.predict_rel(self.rel_theta)
            pred_matrix_rel = pred_matrix_rel.detach().cpu().numpy()
            pred_matrix_rel_bi = pred_matrix_rel_bi.detach().cpu().numpy()

        # filter out the unreliable prediction
        pred_matrix = np.multiply(pred_matrix, pred_matrix_rel_bi)

        # get the test matrix
        # record_matrix = np.where(train_matrix, 0, 1)
        test_ndmat = test_matrix.toarray()
        print('test entries:{}'.format(len(test_ndmat.nonzero()[0])))
        test_ndmat = np.multiply(pred_matrix_rel_bi, test_ndmat)
        print('filtered test entries:{}'.format(len(test_ndmat.nonzero()[0])))

        # filter out the unpredictable test rating
        record_matrix = np.where(test_ndmat, 1, 0)  # only consider the purchased item to test the preference predict
        pred_matrix = np.multiply(record_matrix, pred_matrix)

        # save evaluating file
        if save_file:
            np.save(PRED_SAVEFILE, pred_matrix)
            np.save(TEST_SAVEFILE, test_ndmat)
            print('SAVE FILE SUCCESS!')

        # evaluating
        top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = recall_precision_f1_hr(pred_matrix,
                                                                                                 pred_matrix_rel,
                                                                                                 test_ndmat,
                                                                                                 train_u_avg=None,
                                                                                                 top_k=topk)

        return top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square
