import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import squestions_train
import sensvecs_pretreat

import numpy as np

EVAL_INTERCAL_SECS = 10
BATCH_SIZE = 5000
NUM = 5


def _get_simple_lstm(rnn_size,layer_size):

    lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
    return tf.contrib.rnn.MultiRNNCell(lstm_layers)


def evaluate(test):
    with tf.Graph().as_default() as g:

        input_x1 = tf.placeholder(tf.float32,[1, None, sensvecs_pretreat.VEC_DIM])
        input_x2 = tf.placeholder(tf.float32, [1, None, sensvecs_pretreat.VEC_DIM])
        targets = tf.placeholder(tf.float32,[1, squestions_train.NUM_LABELS])
        x1_len = tf.placeholder(tf.int32, [None])
        x2_len = tf.placeholder(tf.int32, [None])

        with tf.variable_scope('input1'):
            vec1 = _get_simple_lstm(squestions_train.RNN_SIZE, squestions_train.LAYER_SIZE)
            _, state1 = tf.nn.dynamic_rnn(vec1, input_x1, dtype=tf.float32)

        with tf.variable_scope('input2'):
            vec2 = _get_simple_lstm(squestions_train.RNN_SIZE, squestions_train.LAYER_SIZE)
            _, state2 = tf.nn.dynamic_rnn(vec2, input_x2, dtype=tf.float32)

        concation = tf.concat([state1[1][1], state2[1][1]], axis=1)
        logits = tf.layers.dense(concation, 2, activation=tf.nn.softmax)

        logits_norm = tf.sqrt(tf.reduce_sum(tf.square(logits)))
        targets_norm = tf.sqrt(tf.reduce_sum(tf.square(logits)))
        logits_targets = tf.reduce_sum(tf.multiply(logits, targets))
        s_rate = tf.divide(logits_targets, tf.multiply(logits_norm, targets_norm))

        saver = tf.train.Saver()
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(squestions_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('\\')[-1].split('-')[-1]
                    s_list = []
                    for i in range(len(test)-1):
                        accuracy_score = sess.run(s_rate, feed_dict={input_x1: [test[0]], input_x2:[test[i]],})
                        s_list.append(accuracy_score)
                    kind = s_list.index(max(s_list))//NUM
                    print(max(s_list),kind)
                else:
                    print("找不到文件")
                    return
                time.sleep(EVAL_INTERCAL_SECS)



def main(arg = None):
    test = sensvecs_pretreat.test_sqsvec
    evaluate(test)


if __name__ == "__main__":
    tf.app.run()

