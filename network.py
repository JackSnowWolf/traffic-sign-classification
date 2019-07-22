import tensorflow as tf
import numpy as np


class traffic_sign_network():
    def __init__(self, phase="train", num_classes=43):
        self.phase = phase.upper()
        self.num_classes = num_classes
        self.layers = dict()

    def is_train(self):
        if self.phase == "TRAIN":
            return True
        elif self.phase == "TEST":
            return False
        else:
            raise ValueError("Not a valid phase")

    def inference(self, input_data):
        """
        inference
        :param input_data:
        :return: return prediction label and raw output
        """
        with tf.variable_scope(name_or_scope="cnn"):
            output_layer = self.forward(input_data)
            pred = tf.argmax(tf.nn.softmax(output_layer), axis=1, name="pred")
            return pred, output_layer

    def loss(self, labels, input_data):
        """
        compute and loss and do inference
        :param labels: ground truth lable
        :param input_data: input data [batch x num_cells]
        :return: loss and prediction label
        """

        pred, out = self.inference(input_data)
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, out), name="loss") + \
               tf.losses.get_regularization_loss()
        return loss, pred

    def conv_layer(self, input_data, out_dims, name):
        """
        traditional convolution layer unit
        :param input_data:
        :param out_dims:
        :param name:
        :return:
        """

        with tf.variable_scope(name_or_scope=name):
            [_, _, _, channel_num] = input_data.get_shape().as_list()
            w = tf.get_variable("w", [3, 3, channel_num, out_dims],
                                initializer=tf.contrib.layers.variance_scaling_initializer(),
                                trainable=self.is_train())
            conv = tf.nn.conv2d(input_data, w, [1, 1, 1, 1], "SAME", name="conv")
            bn = tf.contrib.layers.batch_norm(conv, scope="bn", trainable=self.is_train())
            relu = tf.nn.relu(bn, name="relu")
        return relu

    def forward(self, input_data, reg_const=0.001):
        """
        forward process
        :param input_data: input_data [batch x height x width x channel]
        :param reg_const: regularization constant
        :return: output result [batch x num_classes]
        """
        [batch_num, _, _, channel_num] = input_data.get_shape().as_list()

        # conv1
        conv1 = self.conv_layer(input_data, 32, "conv1")
        self.layers["conv1"] = conv1
        maxpool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="maxpool1")

        # conv2
        conv2 = self.conv_layer(maxpool1, 64, "conv2")
        self.layers["conv2"] = conv2
        maxpool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="maxpool2")

        # conv3
        conv3 = self.conv_layer(maxpool2, 128, "conv3")
        self.layers["conv3"] = conv3
        # conv4
        conv4 = self.conv_layer(conv3, 128, "conv4")
        self.layers["conv4"] = conv4
        # fully connection
        shape = conv4.get_shape().as_list()[1:]
        before_fc = tf.reshape(conv4, [-1, int(np.prod(shape))])

        fc1 = tf.layers.dense(before_fc, 1024, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              name="fc1", trainable=self.is_train(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_const))
        # fc2 = tf.layers.dense(fc1, 1024, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        #                       name="fc2", trainable=self.is_train())
        fc2 = tf.layers.dense(fc1, self.num_classes,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              name="fc2", trainable=self.is_train(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_const))
        return fc2
