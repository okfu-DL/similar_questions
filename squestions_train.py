import tensorflow as tf
import os
import sensvecs_pretreat

NUM_COL = 6
BATCH_SIZE = 216
VEC_DIM = 128
NUM_LABELS = 2
RNN_SIZE = 256
LAYER_SIZE = 2
GRAD_CLIP = 5
NUM_EPOCH = 200
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


class SQModel(object):
    def __init__(self, is_training, rnn_size, layer_size, grad_clip):

        self.input_x1 = tf.placeholder(tf.float32,[None, None, VEC_DIM])
        self.input_x2 = tf.placeholder(tf.float32,[None, None, VEC_DIM])
        self.targets = tf.placeholder(tf.float32,[None, NUM_LABELS])
        self.x1_len = tf.placeholder(tf.int32, [None])
        self.x2_len = tf.placeholder(tf.int32, [None])
        with tf.variable_scope('input1'):
            vec1 = self._get_simple_lstm(rnn_size, layer_size)
            _, state1 = tf.nn.dynamic_rnn(vec1, self.input_x1, sequence_length=self.x1_len, dtype=tf.float32)

        with tf.variable_scope('input2'):
            vec2 = self._get_simple_lstm(rnn_size, layer_size)
            _, state2 = tf.nn.dynamic_rnn(vec2, self.input_x2, sequence_length=self.x2_len, dtype=tf.float32)

        with tf.variable_scope('concation', reuse=tf.AUTO_REUSE):
            concation = tf.concat( [state1[1][1],state2[1][1]], axis=1)

        with tf.variable_scope('dense'):
            weights = tf.get_variable("weights", [512, 2], initializer= tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases",[2],initializer=tf.constant_initializer(0.1))
            logits = tf.nn.softmax(tf.matmul(concation, weights)+biases)
            # logits = tf.layers.dense(concation, 2, activation=tf.nn.softmax)

        if is_training:

            self.cost = -tf.reduce_mean(tf.reduce_sum(self.targets*tf.log(logits)))
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            logits_norm = tf.sqrt(tf.reduce_sum(tf.square(logits), axis=1))
            targets_norm = tf.sqrt(tf.reduce_sum(tf.square(self.targets), axis=1))
            logits_targets = tf.reduce_sum(tf.multiply(logits, self.targets), axis=1)
            loss = tf.divide(logits_targets, tf.multiply(logits_norm, targets_norm))
            self.cost = tf.reduce_mean(loss)
            return


    def _get_simple_lstm(self, rnn_size, layer_size):

        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


def run_epoch(session, model, data, train_op, is_training):

    total_costs = 0.0

    if is_training:
        for i in range(6):
            start = (i * BATCH_SIZE) % len(data.sens1)
            end = min(start+BATCH_SIZE, len(data.sens1))
            x1_len = sensvecs_pretreat.sens1_length[start:end]
            x2_len = sensvecs_pretreat.sens2_length[start:end]
            cost, _ = session.run([model.cost, train_op],
                                  {model.input_x1: data.sens1[start:end], model.input_x2:data.sens2[start:end],
                                   model.targets: data.labels[start:end],model.x1_len:x1_len,model.x2_len:x2_len })
            total_costs += cost
        print("平均loss:%s" % total_costs/6)
        return total_costs
    else:
        s_list = []
        for i in range(1,len(data)):
            accuracy_score = session.run(model.cost, feed_dict={model.input_x1: [data[0]], model.input_x2: [data[i]],
                                                            model.targets:[[1.0, 0.0]],model.x1_len: [len(data[0])],model.x2_len:[len(data[i])]})
            s_list.append(accuracy_score)
        kind = s_list.index(max(s_list)) // NUM_COL
        print(s_list, max(s_list), kind)
        return kind


def train():

    train_data = sensvecs_pretreat.train_sqsvec

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = SQModel(True, RNN_SIZE, LAYER_SIZE, GRAD_CLIP)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(NUM_EPOCH):
            print("第"+str(i+1)+"次迭代")
            run_epoch(sess, train_model, train_data, train_model.train_op, True)
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))



def test1():

    test_data = sensvecs_pretreat.test_sqsvec

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=tf.AUTO_REUSE, initializer=initializer):
        eval_model = SQModel(False, RNN_SIZE, LAYER_SIZE, GRAD_CLIP)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            same_kind = run_epoch(sess, eval_model, test_data, tf.no_op(), False)
            print("最相似种类为%d" % same_kind)
        else:
            print("找不到文件")
            return


def main(_):

  # train()
  test1()


if __name__ == "__main__":
    tf.app.run()






