import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from CapsNet import CapsNet


def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        loss = cfg.results + '/loss.csv'
        train_acc = cfg.results + '/train_acc.csv'
        val_acc = cfg.results + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')

        return fd_train_acc, fd_loss, fd_val_acc
    else:
        test_acc = cfg.results + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')

        return fd_test_acc


def train(model, num_label, supervisor):
    train_X, train_Y, train_batch_num, val_X, val_Y, val_batch_num = load_data(cfg.dataset, cfg.batch_size, is_training=True)

    Y = val_Y[:val_batch_num * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss, fd_val_acc = save_to()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    # In `supervisor.managed_session` block, all variables in the graph have been initialized. In addition,
    # a few services have been started to checkpoint the model and add summaries to the event log.
    with supervisor.managed_session(config=config) as sess:
        print("\nNote: all of results will be saved to directory: " + cfg.results)

        for epoch in xrange(cfg.epoch):
            print('Training for epoch ' + str(epoch) + '/' + str(cfg.epoch) + ':')

            # The supervisor is notified of any exception raised by one of the services.
            # After an exception is raised, `should_stop()` returns `True`.  In that case
            # the training loop should also stop.  This is why the training loop has to
            # check for `sv.should_stop()`.
            if supervisor.should_stop():
                print('supervisor stoped!')
                break

            for step in tqdm(xrange(train_batch_num), total=train_batch_num, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])

                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(model.accuracy, {model.X: val_X[start:end], model.labels: val_Y[start:end]})
                        val_acc += acc

                    val_acc = val_acc / (cfg.batch_size * val_batch_num)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()

            if (epoch + 1) % cfg.save_freq == 0:
            supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    test_X, test_Y, test_batch_num = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    fd_test_acc = save_to()

    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(test_batch_num), total=test_batch_num, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: test_X[start:end], model.labels: test_Y[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * test_batch_num)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + cfg.results + '/test_accuracy.txt')


def main(_):
    tf.logging.info(' Loading Graph...')
    num_label = 10
    model = CapsNet()
    tf.logging.info(' Graph loaded')

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model, num_label, sv)
        tf.logging.info('Training done')
    else:
        evaluation(model, num_label, sv)

if __name__ == "__main__":
    # def run(main=None, argv=None):
    # Runs the program with an optional 'main' function and 'argv' list.
    # main = main or sys.modules['__main__'].main
    tf.app.run()
