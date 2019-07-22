import os
import time
import numpy as np
from optparse import OptionParser
import tensorflow as tf
import glog as logger
import cv2
import data_loader
import network


def my_parser():
    """
    parse arguments
    :return: options
    """
    parser = OptionParser()

    parser.add_option("-i", "--images", action="store", dest="images",
                      type="string", default=None,
                      help="set image path or directory")
    parser.add_option("-w", "--weight", action="store", dest="weight_path",
                      type="string",
                      default=None,
                      help="path to model weight")
    parser.add_option("-v", "--visualize", action="store_true", dest="visualize",
                      default=False,
                      help="switch to visualize network")

    options, _ = parser.parse_args()
    return options


def compute_acc(preds, labels):
    acc = np.size(np.where(preds == labels)) / preds.size
    return acc


def visualize_feature(layer_value, layer_num):
    out_dir = "demo/visualization"
    for i in range(len(layer_value)):
        shape = layer_value[i].shape
        # layer_value[i] = layer_value[i] / np.max(layer_value) * 255.0
        features = []
        for j in range(shape[-1]):
            feature = layer_value[i][:, :, j]
            feature = np.expand_dims(feature, -1)
            feature = feature / np.max(feature) * 255.0
            out_im = cv2.resize(feature, (32, 32))
            features.append(out_im)
            out_name = "im_%02d_layer_%01d_%03d.jpg" % (i + 1, layer_num, j)
            # cv2.imwrite(os.path.join(out_dir, out_name), out_im)
        features = np.stack(features[0:16])
        features = np.reshape(features, (4, 4, 32, 32, 1))
        features = np.swapaxes(features, 1, 2)
        features = np.reshape(features, (4 * 32, 4 * 32, 1))
        out_name = "im_%02d_layer_%01d.jpg" % (i + 1, layer_num)
        cv2.imwrite(os.path.join(out_dir, out_name), features)


def traffic_sign_demo_with_test_images(weight_path, visualize=False):
    # prepare output dir
    if os.path.exists("demo/test"):
        os.system("rm -rf demo/test/*")
    else:
        os.mkdir("demo/test")
    if os.path.exists("demo/visualization"):
        os.system("rm -rf demo/visualization/*")
    else:
        os.mkdir("demo/visualization")
    if visualize:
        out_dir = "demo/visualization"
        if os.path.exists(out_dir):
            os.system("rm -rf %s/*" % out_dir)
        else:
            os.mkdir(out_dir)

    batch_size = 10
    # load data
    test_data, test_labels = data_loader.x_test, data_loader.y_test
    assert len(test_data) == len(test_labels)

    # random pick 10 images
    batch_index = np.random.choice(np.arange(len(test_labels)), size=10, replace=False)
    labels_batch = test_labels[batch_index]
    image_batch = test_data[batch_index]
    # save images
    for i in range(len(image_batch)):
        img_bgr = cv2.cvtColor(image_batch[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite("demo/test/img_%02d_%02d.jpg" % (i, labels_batch[i]), img_bgr)

    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name="input")
    traffic_sign_net = network.traffic_sign_network(phase="test", num_classes=43)

    pred, raw = traffic_sign_net.inference(input_data=input_layer)
    layers = traffic_sign_net.layers

    saver = tf.train.Saver()

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=sess_config)

    # start training
    with sess.as_default():
        if weight_path is None:
            raise ValueError("weight path doesn't configured")

        logger.info("Restore model from {:s}".format(weight_path))
        saver.restore(sess=sess, save_path=weight_path)
        if not visualize:
            pred_label, raw_label = sess.run([pred, raw], feed_dict={input_layer: image_batch})
        else:
            out_items = [pred, raw]
            for name_layer, layer in layers.items():
                out_items.append(layer)
            out_values = sess.run(out_items, feed_dict={input_layer: image_batch})
            pred_label = out_values[0]
            raw_label = out_values[1]
            layers_value = out_values[2:]
            for layer_num, layer_v in enumerate(layers_value):
                visualize_feature(layer_v, layer_num + 1)

        for i in range(len(pred_label)):
            max_p_ls = np.argsort(-raw_label[i])[0:5]
            print("image %02d" % i, end="\t")
            print("label %02d" % labels_batch[i], end="\t")
            print("predict %02d" % pred_label[i])
            print("top 5 softmax probabilities:", end="\t")
            print(max_p_ls)

        test_acc = compute_acc(preds=pred_label, labels=labels_batch)
        acc = test_acc * 100
        logger.info("test acc: {:.2f}%%".format(acc))


def traffic_sign_demo_with_show_images(images, weight_path):
    # prepare output dir

    if os.path.exists("demo/visualization"):
        os.system("rm -rf demo/visualization/*")
    else:
        os.mkdir("demo/visualization")

    # load demo images
    images_ls = []
    image_batch = []
    if os.path.isdir(images):
        images_list = os.listdir(images)
        images_list.sort()
        images_ls = [os.path.join("demo/show/", img_name) for img_name in images_list]
    else:
        images_ls.append(images)

    for img_name in images_ls:
        im = cv2.imread(img_name)
        im_t = cv2.cvtColor(cv2.resize(im, (32, 32)), cv2.COLOR_BGR2RGB)
        image_batch.append(im_t)

    # # save images
    # for i in range(len(image_batch)):
    #     img_bgr = cv2.cvtColor(image_batch[i], cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("demo/out/img_show_%02d.jpg" % (i + 1), img_bgr)

    # set configuration
    batch_size = len(image_batch)
    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3], name="input")
    traffic_sign_net = network.traffic_sign_network(phase="test", num_classes=43)

    pred, raw = traffic_sign_net.inference(input_data=input_layer)

    saver = tf.train.Saver()

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=sess_config)

    # start training
    with sess.as_default():
        if weight_path is None:
            raise ValueError("weight path doesn't configured")

        logger.info("Restore model from {:s}".format(weight_path))
        saver.restore(sess=sess, save_path=weight_path)

        pred_label, raw_label = sess.run([pred, raw], feed_dict={input_layer: image_batch})
        for i in range(len(pred_label)):
            max_p_ls = np.argsort(-raw_label[i])[0:5]
            print("image %02d" % (i + 1), end="\t")
            print("predict %02d" % pred_label[i])
            print("top 5 softmax probabilities:", end="\t")
            print(max_p_ls)


if __name__ == '__main__':
    opt_init = my_parser()
    if opt_init.images is None:
        traffic_sign_demo_with_test_images(opt_init.weight_path, opt_init.visualize)
    else:
        traffic_sign_demo_with_show_images(opt_init.images, opt_init.weight_path)
    logger.info("Done!")
