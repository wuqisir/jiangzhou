import torch
import torch.optim as optim

from model.LightGCN import LightGCN
from model.BPRMF import BPRMF
from model.LGNGuard import LGNGuard
from model.NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
from utility.parser import args

import warnings
warnings.filterwarnings('ignore')
from time import time

if __name__ == '__main__':
    
    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat() # 'norm_adj' for LightGCN; 'mean_adj' for NGCF
    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    # path to save model
    weight_file = f"clean_{args.dataset}_{args.model}.pkl"

    model = NGCF(data_generator.n_users,
                data_generator.n_items,
                mean_adj,
                args).to(args.device)

    print(f"start train {args.model}")

    t0 = time()
    """
    *********************************************************
    Train.
    """
    users_to_test = list(data_generator.test_set.keys())
    ret0, dict_cdf, rmse, ppr, rating, user = test(model, users_to_test, drop_flag=False)
    print(ret0)
    print(dict_cdf)
    print(rmse, ppr)

    cur_best_pre_0, stopping_step = 0, 0

    # if args.model_type == 'NCF':
    #     optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    rating_save = 0

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger, f1_loger, rmse_loger, ppr_loger, cdf_loger = [], [], [], [], [], [], [], [], []
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            batch_loss, batch_mf_loss, batch_emb_loss = getattr(model, f"create_{args.loss_type}_loss")(users, pos_items, neg_items, drop_flag=args.node_dropout_flag)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % \
                (epoch, time() - t1, loss, mf_loss, emb_loss)
        print(perf_str)
        if (epoch+1)% 2 == 0:
            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            ret, dict_cdf, rmse, ppr, rating, user_save = test(model, users_to_test, drop_flag=False)
            # print(ret0)
            t3 = time()
            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            f1_loger.append(ret['f1'])
            rmse_loger.append(rmse)
            ppr_loger.append(ppr)
            cdf_loger.append(dict_cdf)

            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f]' % \
                        (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss)
            print(perf_str)
            print(ret)
            print(dict_cdf)
            print(rmse, ppr)
            # *********************************************************
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['ndcg'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step= 100)
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            # if ret['ndcg'] < 0.834:
            #     rating_save = ret['ndcg']
            #     np.save(f"./dataset/{args.dataset}/{args.dataset}_result", rating)
            #     np.save(f"./dataset/{args.dataset}/{args.dataset}_user", user_save)
            #     torch.save(model.state_dict(), weight_file)
            if should_stop == True:
                rating_save = ret['ndcg']
                np.save(f"./dataset/{args.dataset}/{args.dataset}_result", rating)
                np.save(f"./dataset/{args.dataset}/{args.dataset}_user", user_save)
                torch.save(model.state_dict(), weight_file)
                break


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    f1 = np.array(f1_loger)

    best_ndcg_0 = min(ndcgs[:, 0])
    idx = list(ndcgs[:, 0]).index(best_ndcg_0)
    save = [recs[idx, 0], pres[idx, 0], ndcgs[idx, 0], f1[idx, 0], ppr_loger[idx], rmse_loger[idx]]
    cdf = cdf_loger[idx]
    list = []
    for key in cdf.keys():
        tmp = [key, cdf[key]]
        list.append(tmp)
    print(save)
    df = pd.DataFrame([save], columns=['recall', 'precision','ndcg', 'f1', 'ppr', 'rmse'])
    df.to_csv(f"./dataset/{args.dataset}/result.csv")
    df = pd.DataFrame(list)
    df.to_csv(f"./dataset/{args.dataset}/cdf_result", header=None)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    # torch.save(model.state_dict(), weight_file)
    print("model saved in ", weight_file)
