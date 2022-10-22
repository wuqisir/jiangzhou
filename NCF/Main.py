import os, sys, time
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math

import NCF_input as NCF_input
import NCF as NCF
import Metrics as Metrics

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128, 'size of mini-batch.')
# tf.app.flags.DEFINE_integer('negative_num', 4, 'number of negative samples.')
# tf.app.flags.DEFINE_integer('test_neg', 99, 'number of negative samples for test.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('epochs', 1000, 'the number of epochs.')
tf.app.flags.DEFINE_integer('topk', 5, 'topk for evaluation.')
tf.app.flags.DEFINE_string('optim', 'Adam', 'the optimization method.')
tf.app.flags.DEFINE_string('initializer', 'original', 'the initializer method.')
tf.app.flags.DEFINE_string('loss_func', 'cross_entrop', 'the loss function.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')
# tf.app.flags.DEFINE_string('model_dir', 'model/', 'the dir fro saving model.')
tf.app.flags.DEFINE_float('regularizer', 1e-6, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.0005, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.2, 'dropout rate.')
tf.compat.v1.set_random_seed(0)
PRED_SAVEFILE = './pred_array_NCF'
TEST_SAVEFILE = './test_array_NCF'


def train(train_data, test_data, user_size, item_size, test_features, test_labels):
    # config = tf.ConfigProto(allow_growth=True,
    #                         allow_soft_placement=True)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        iterator = tf.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_data),
                                                   tf.compat.v1.data.get_output_shapes(train_data))

        model = NCF.NCF(FLAGS.embedding_size, user_size, item_size, FLAGS.lr,
                        FLAGS.optim, FLAGS.initializer, FLAGS.loss_func, FLAGS.activation,
                        FLAGS.regularizer, iterator, FLAGS.topk, FLAGS.dropout, is_training=True)
        model.build()

        # ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        # if ckpt:
        #     print('Reading model parameters from %s' % ckpt.model_checkpoint_path)
        #     model.saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     print('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())

        count = 0
        for epoch in range(FLAGS.epochs):
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            model.get_data()
            start_time = time.time()

            try:
                while True:
                    model.step(sess, count)
                    count += 1
            except tf.errors.OutOfRangeError:
                print('training_time:{}'.format(time.time() - start_time))
                print('Epoch %d training' % epoch + 'Took:' + time.strftime('%H: %M: %S',
                                                                            time.gmtime(time.time() - start_time)))
            sess.run(model.iterator.make_initializer(test_data))
            model.is_training = False
            model.get_data()
            start_time = time.time()

            PRED = []
            # prediction, label = model.step(sess, None)
            try:
                while True:
                    # prediction, label = model.step(sess, None)
                    result = model.step(sess, None)
                    # label = int(label[0])
                    PRED.append(result)
            except tf.errors.OutOfRangeError:
                # Target RMSE
                prediction = Metrics.pred_value(PRED)
                rmse = Metrics.rmse(prediction, test_labels)
                print(f'Epoch:{epoch}; RMSE:{rmse}')
                target_rmse = 1.287
                if (target_rmse - 0.05) <= rmse <= (target_rmse + 0.05):
                    np.save(PRED_SAVEFILE + '_' + str(epoch), prediction)
                    np.save(TEST_SAVEFILE + '_' + str(epoch), test_labels)
                    print(f'SAVE EPOCH {epoch} FILE SUCCESS!')

                if (epoch % 10) == 0:
                    save_flag = True if epoch == (FLAGS.epochs - 1) else False
                    prediction = Metrics.pred_value(PRED)
                    rmse = Metrics.rmse(prediction, test_labels)
                    rounded_rmse = Metrics.rounded_rmse(prediction, test_labels)
                    # p, r, f, ndcg = Metrics.f1_value(prediction, test_labels, test_features, 5)
                    PRE, REC, F1, HR, NDCG=Metrics.f1_value(prediction, test_labels, test_features, 5)
                    ppr, ks = Metrics.rating_ability(prediction, test_labels, test_features)
                    r_square = Metrics.R_square(prediction, test_labels)
                    print('Epoch %d testing' % epoch + 'Took:' + time.strftime('%H: %M: %S',
                                                                               time.gmtime(time.time() - start_time)))
                    # print(
                    #     'RMSE is %.3f, PRE is %.3f, REC is %.3f, F1 is %.3f, NDCG is %.3f, PPR is %.3f, KKS is %.3f, R_SQUARE is %.3f' % (
                    #         rmse, p, r, f, ndcg, ppr, ks, r_square))
                    print(HR,NDCG)
                    print('ROUNDED RMSE is %.3f' % (rounded_rmse))
                    if save_flag:
                        np.save(PRED_SAVEFILE, prediction)
                        np.save(TEST_SAVEFILE, test_labels)
                        print('SAVE FILE SUCCESS!')

        # checkpoint_path = os.path.join(FLAGS.model_dir, 'NCF.ckpt')
        # model.saver.save(sess, checkpoint_path)


def main():
    # ((train_features, train_labels),
    #  (test_features, test_labels),
    #  (user_size, item_size),
    #  (user_bought, user_negative)) = NCF_input.load_data()
    ((train_features, train_labels),
     (test_features, test_labels),
     (user_size, item_size)) = NCF_input.load_data()
    # print(train_features[:10])

    # train_data = NCF_input.train_input_fn(train_features, train_labels, FLAGS.batch_size, user_negative,
    #                                       FLAGS.negative_num)
    train_data = NCF_input.train_input_fn(train_features, train_labels, FLAGS.batch_size)

    # test_data = NCF_input.eval_input_fn(test_features, test_labels,
    #                                     user_negative, FLAGS.test_neg)
    test_data = NCF_input.eval_input_fn(test_features, test_labels)

    train(train_data, test_data, user_size, item_size, test_features, test_labels)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("Total time:", time_end - time_start)
