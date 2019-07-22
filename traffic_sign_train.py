import os
import tensorflow as tf
from optparse import OptionParser
import time
import numpy as np
import glog as logger
import data_loader
import data_feeder
import network


def compute_acc(preds, labels):
    acc = np.size(np.where(preds == labels)) / preds.size
    return acc


def my_parser():
    """
    parse arguments
    :return: options
    """
    parser = OptionParser()
    parser.add_option("--lr", "--learning_rate", action="store",
                      dest="learning_rate",
                      type="float", default=0.0005,
                      help="set learning_rate")
    parser.add_option("--batch_size", action="store", dest="batch_size",
                      type="int", default=32,
                      help="set batch size")
    parser.add_option("-w", "--weight", action="store", dest="weight_path",
                      type="string",
                      default=None,
                      help="path to pretrain weight or previous weight")
    parser.add_option("--tboard", action="store", dest="tboard",
                      type="string", default="tboard",
                      help="set tensor board log directory")
    parser.add_option("-n", "--steps", action="store", dest="steps",
                      default=10000, type="int", help="set train steps")
    parser.add_option("-s", "--save_path", action="store", dest="save_path",
                      type="string", default="model",
                      help="set model save path")

    options, _ = parser.parse_args()
    return options


def traffic_sign_train(lr, batch_size, weight_path, train_epochs, tboard_dir, save_path):
    # load data
    x_train, y_train = data_loader.x_train, data_loader.y_train
    x_validation, y_validation = data_loader.x_validation, data_loader.y_validation
    assert len(x_train) == len(y_train)
    data_feed = data_feeder.DataFeeder(x_train, y_train, batch_size=batch_size)
    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name="input")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="labels")

    traffic_sign_net = network.traffic_sign_network(phase="TRAIN", num_classes=43)
    loss, pred = traffic_sign_net.loss(labels=labels, input_data=input_layer)

    # set learning rate
    global_step = tf.Variable(0, name="global_step", trainable=False)

    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grad = optimizer.compute_gradients(loss=loss)

    apply_grad_op = optimizer.apply_gradients(grad, global_step=global_step)

    # set tensorflow summary
    tboard_save_path = tboard_dir
    os.makedirs(tboard_save_path, exist_ok=True)
    summary = tf.summary.FileWriter(tboard_save_path)

    train_loss_scalar = tf.summary.scalar(name="train_loss", tensor=loss)
    learning_rate_scalar = tf.summary.scalar(name="learning_rate", tensor=learning_rate)
    train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # train_merge_summary_op = tf.summary.merge([train_loss_scalar, learning_rate_scalar], train_summary_op_updates)
    train_merge_summary_op = tf.summary.merge_all()

    # set saver
    os.makedirs(save_path, exist_ok=True)

    saver = tf.train.Saver(max_to_keep=10)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    sess = tf.Session(config=sess_config)
    summary.add_graph(sess.graph)

    # start training
    with sess.as_default():
        epoch = 0
        if weight_path is None:
            logger.info("Training from scratch")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info("Restore model from {:s}".format(weight_path))
            saver.restore(sess=sess, save_path=weight_path)
        train_loss_list = []
        train_acc_list = []
        while epoch < train_epochs:
            epoch += 1
            data_batch, labels_batch = data_feed.next_batch()
            _, train_loss, pred_label, train_merge_summary_value = sess.run(
                [apply_grad_op, loss, pred, train_merge_summary_op],
                feed_dict={input_layer: data_batch,
                           labels: labels_batch})
            acc = compute_acc(preds=pred_label, labels=labels_batch) * 100
            train_loss_list.append(train_loss)
            train_acc_list.append(acc)
            if epoch % 100 == 0:
                acc = sum(train_acc_list) / len(train_acc_list)
                train_loss = sum(train_loss_list) / len(train_loss_list)
                train_acc_list = []
                train_loss_list = []
                logger.info("epoch: {:d}\ttrain loss: {:.3f}\ttrain acc: {:.2f}%%".format(epoch, train_loss, acc))
                summary.add_summary(summary=train_merge_summary_value, global_step=epoch)
            if epoch % 500 == 0:
                # validation
                val_acc_list = []
                val_loss_list = []
                batch_num = int(np.floor(y_validation.size / batch_size))
                for i in range(batch_num):
                    data_batch = x_validation[i * batch_size:(i + 1) * batch_size]
                    labels_batch = y_validation[i * batch_size:(i + 1) * batch_size]
                    val_loss, pred_label = sess.run([loss, pred], feed_dict={input_layer: data_batch,
                                                                             labels: labels_batch})
                    val_acc = compute_acc(preds=pred_label, labels=labels_batch)
                    val_acc_list.append(val_acc)
                    val_loss_list.append(val_loss)
                acc = sum(val_acc_list) / len(val_acc_list) * 100
                val_loss = sum(val_loss_list) / len(val_loss_list)
                logger.info("epoch: {:d}\tval loss: {:.3f}\tval acc: {:.2f}%%".format(epoch, val_loss, acc))

                model_name = 'traffic_sign_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
                model_save_path = os.path.join(save_path, model_name)
                saver.save(sess=sess, save_path=model_save_path)
        model_name = 'traffic_sign_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
        model_save_path = os.path.join(save_path, model_name)
        saver.save(sess=sess, save_path=model_save_path)


if __name__ == '__main__':
    init_opt = my_parser()
    traffic_sign_train(lr=init_opt.learning_rate, batch_size=init_opt.batch_size, weight_path=init_opt.weight_path,
                       train_epochs=init_opt.steps, tboard_dir=init_opt.tboard, save_path=init_opt.save_path)

    logger.info("done!")
