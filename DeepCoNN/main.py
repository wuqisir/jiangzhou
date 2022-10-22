# -*- encoding: utf-8 -*-
import math
import random
import time

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from evaluate import recall_precision_f1_hr_tensor, recall_precision_f1_hr
from evaluate import recall_precision_f1_hr

import config
import models
from dataset import ReviewData
from framework import Model


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)  # args passed in will replace the default ones

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_mat = list_to_matrix(val_data.x, opt.user_num, opt.item_num)  # get the val mat for evaluating
    # val_data_tensor = torch.from_numpy(val_data_mat)
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")
    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"training start: {now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            print('\r' + 'epoch:' + str(epoch) + str(idx) + '/' + str(len(train_data_loader)), end='', flush=True)
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
                # loss = torch.sqrt(mse_loss)
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.print_opt)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss
        print(f"training end: {now()}  Epoch {epoch}...")
        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")

        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            model.save(name=opt.dataset, opt=opt.print_opt)
            min_loss = val_loss
            print("model save")
        if val_mse < best_res:
            best_res = val_mse
        print("*" * 30)

        # Target RMSE
        if opt.target_rmse_flag:
            target_rmse = opt.target_rmse
            current_rmse = math.sqrt(val_mse)
            if (target_rmse - 0.05) <= current_rmse <= (target_rmse + 0.05):
                model.eval()
                output = []
                test_ratings = []
                with torch.no_grad():
                    for idx, (test_data, scores) in enumerate(val_data_loader):
                        if opt.use_gpu:
                            scores = torch.FloatTensor(scores).cuda()
                        else:
                            scores = torch.FloatTensor(scores)
                        test_data = unpack_input(opt, test_data)
                        output_per_batch = model(test_data).detach().cpu().numpy()
                        # output_per_batch = model(test_data).numpy()
                        # output_per_batch = model(test_data)
                        output.extend(output_per_batch)
                        test_ratings.extend(scores.detach().cpu().numpy())

                # # get the prediction matrix
                # pred_matrix = np.zeros_like(val_data_mat)
                # # pred_matrix = torch.zeros_like(test_matrix)
                # # record_matrix = test_matrix.astype(bool) - 1  # mark the nonzero entries
                # index_row = val_data_loader.dataset.data[:, 0]
                # index_col = val_data_loader.dataset.data[:, 1]
                # pred_indexes = tuple([index_row, index_col])
                # # pred_indexes = np.where(test_matrix > 0)
                # # pred_indexes = torch.where(test_matrix > 0)
                # # pred_indexes = (pred_indexes[0][:100], pred_indexes[1][:100])
                # # pred_matrix[pred_indexes] = torch.tensor(output)
                # pred_matrix[pred_indexes] = output
                PRED_SAVEFILE = f'./pred_array_{str(opt.model)}_{epoch}_{current_rmse}'
                TEST_SAVEFILE = f'./test_array_{str(opt.model)}_{epoch}_{current_rmse}'
                np.save(PRED_SAVEFILE, np.array(output))
                np.save(TEST_SAVEFILE, np.array(test_ratings))
                print('SAVE FILE SUCCESS!')

        if (epoch % 10000) == 0:
            save_flag = True if epoch == (opt.num_epochs - 1) else False
            top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = eval_metrics(model, val_data_mat,
                                                                                           val_data_loader,
                                                                                           opt,
                                                                                           topk=5,
                                                                                           save_file=save_flag)
            print(
                'e:{}, top_k:{}, precision:{}, recall:{}, f1_score:{}, ndcg:{}, hr:{}, ppr:{}, ks:{}, r_square:{}'.format(
                    epoch,
                    top_k,
                    precision,
                    recall,
                    f1_score,
                    ndcg, hr,
                    ppr, ks, r_square))

    print("----" * 20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----" * 20)


def test(**kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    print("path:{}".format(opt.pth_path))
    assert (len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, data_loader, opt, save_file=False):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    test_ratings = []
    predictions = []
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)
            mse_loss = torch.sum((output - scores) ** 2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output - scores))
            total_maeloss += mae_loss.item()

            test_ratings.extend(scores.detach().cpu().numpy())
            predictions.extend(output.detach().cpu().numpy())

    if save_file:
        print(test_ratings)
        PRED_SAVEFILE = f'./pred_array_{str(opt.model)}'
        TEST_SAVEFILE = f'./test_array_{str(opt.model)}'
        np.save(PRED_SAVEFILE, np.array(predictions))
        np.save(TEST_SAVEFILE, np.array(test_ratings))
        print('SAVE FILE SUCCESS!')

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation results: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    model.train()
    return total_loss, mse, mae


def compute_rmse_array(prediction, rating, clip=False, rmin=1, rmax=5):
    # np.save('pred_array_NCF', np.array(prediction))
    # np.save('test_array_NCF', np.array(rating))
    sum_error = 0
    count = 0
    if clip:
        prediction = np.clip(prediction, rmin, rmax)
    for i in range(len(prediction)):
        error = math.fabs(prediction[i] - rating[i])
        sum_error += error * error
        count += 1
    RMSE = math.sqrt(sum_error / count)
    return RMSE


def eval_metrics(model, test_matrix, data_loader, opt, topk=5, save_file=False):
    model.eval()
    output = []

    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            test_data = unpack_input(opt, test_data)
            output_per_batch = model(test_data).detach().cpu().numpy()
            # output_per_batch = model(test_data).numpy()
            # output_per_batch = model(test_data)
            output.extend(output_per_batch)

    # get the prediction matrix
    pred_matrix = np.zeros_like(test_matrix)
    # pred_matrix = torch.zeros_like(test_matrix)
    # record_matrix = test_matrix.astype(bool) - 1  # mark the nonzero entries
    index_row = data_loader.dataset.data[:, 0]
    index_col = data_loader.dataset.data[:, 1]
    pred_indexes = tuple([index_row, index_col])
    # pred_indexes = np.where(test_matrix > 0)
    # pred_indexes = torch.where(test_matrix > 0)
    # pred_indexes = (pred_indexes[0][:100], pred_indexes[1][:100])
    # pred_matrix[pred_indexes] = torch.tensor(output)
    pred_matrix[pred_indexes] = output

    if save_file:
        PRED_SAVEFILE = f'./pred_mat_{str(opt.model)}'
        TEST_SAVEFILE = f'./test_mat_{str(opt.model)}'
        np.save(PRED_SAVEFILE, pred_matrix)
        np.save(TEST_SAVEFILE, test_matrix)
        print('SAVE FILE SUCCESS!')

    # evaluating
    top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square = recall_precision_f1_hr(pred_matrix, test_matrix,
                                                                                             train_u_avg=None,
                                                                                             top_k=topk)
    # top_k, recall, precision, f1_score, hr, ndcg = recall_precision_f1_hr_tensor(pred_matrix, test_matrix,
    #                                                                              top_k=topk)

    return top_k, recall, precision, f1_score, hr, ndcg, ppr, ks, r_square


def unpack_input(opt, x):
    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id
    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x).to(device), data))
    return data


def list_to_matrix(dataset, maxu, maxi):
    dataMat = np.zeros([maxu, maxi], dtype=np.float64)
    for [u, i], r in dataset:
        dataMat[int(u)][int(i)] = float(r)
    return np.array(dataMat)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fire.Fire()
    # train(model='DeepCoNN')
