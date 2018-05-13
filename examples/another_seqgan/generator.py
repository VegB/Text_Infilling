from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import texar as tx


def embedding_drop(embedding_matrix, keep_prob):
    mask = tf.nn.dropout(tf.ones((embedding_matrix.shape[0], 1)), keep_prob)
    return mask * embedding_matrix


class Generator(tx.modules.ModuleBase):
    def __init__(self, config, word2id, bos, eos, pad, hparams=None):
        tx.ModuleBase.__init__(self, hparams)

        with tf.variable_scope("generator"):
            self.batch_size = config.batch_size
            self.max_seq_length = config.num_steps
            self.vocab_size = len(word2id)
            self.bos_id = bos
            self.eos_id = eos
            self.pad_id = pad
            self.embedding_dim = self.hparams.embedding_dim
            self.num_layers = self.hparams.num_layers
            self.hidden_units = self.hparams.hidden_units
            self.input_dropout = self.hparams.input_dropout
            self.output_dropout = self.hparams.output_dropout
            self.state_dropout = self.hparams.state_dropout
            self.intra_layer_dropout = self.hparams.intra_layer_dropout
            self.embedding_dropout = self.hparams.embedding_dropout
            self.variational_recurrent = self.hparams.variational_recurrent

            self.data_batch = tf.placeholder(dtype=tf.int32, name="data_batch",
                                             shape=[None, self.max_seq_length + 2])
            self.rewards = tf.placeholder(dtype=tf.float32, name='rewards',
                                          shape=[None, self.max_seq_length])

            self.output_layer = \
                tf.layers.Dense(units=self.vocab_size, use_bias=False)
            self.output_layer(tf.ones([1, self.embedding_dim]))

            cell_list = [tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self.hidden_units)]
            for i in range(1, self.num_layers):
                cell_list.append(tf.nn.rnn_cell.DropoutWrapper(
                    cell=tf.nn.rnn_cell.BasicLSTMCell(
                        num_units=self.hidden_units),
                    input_keep_prob=tx.utils.switch_dropout(
                        1. - self.intra_layer_dropout),
                    variational_recurrent=self.variational_recurrent,
                    input_size=self.hidden_units,
                    dtype=tf.float32
                ))
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell(cells=cell_list),
                input_keep_prob=tx.utils.switch_dropout(
                    1. - self.input_dropout),
                output_keep_prob=tx.utils.switch_dropout(
                    1. - self.output_dropout),
                state_keep_prob=tx.utils.switch_dropout(
                    1. - self.state_dropout),
                variational_recurrent=self.variational_recurrent,
                input_size=self.embedding_dim,
                dtype=tf.float32)
            self.decoder = tx.modules.BasicRNNDecoder(
                cell=cell, vocab_size=self.vocab_size)#self.embedding_dim)

            embedding_matrix = tf.transpose(self.output_layer.weights[0])
            self.embedding_matrix = embedding_drop(
                embedding_matrix,
                tx.utils.switch_dropout(1. - self.embedding_dropout))

            initial_state = self.decoder.zero_state(
                batch_size=tf.shape(self.data_batch[:, :-1])[0], dtype=tf.float32)
            self.outputs, final_state, sequence_length = self.decoder(
                inputs=tf.nn.embedding_lookup(self.embedding_matrix, self.data_batch[:, :-1]),
                initial_state=initial_state,
                impute_finished=True,
                decoding_strategy="train_greedy",
                sequence_length=[self.max_seq_length + 1] * self.batch_size)

            # Losses & train ops
            self.teacher_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=self.data_batch[:, 1:],
                logits=self.outputs.logits,
                sequence_length=config.num_steps * tf.ones((self.batch_size,)))

            l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

            # Use global_step to pass epoch, for lr decay
            self.global_step = tf.Variable(0, dtype=tf.int32)
            self.learning_rate = \
                tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=0.,
                beta2=0.999,
                epsilon=1e-9)
            self.train_op = optimizer.minimize(
                self.teacher_loss + config.l2_decay * l2_loss, global_step=self.global_step)

            self.generated_outputs, _, _ = self.decoder(
                decoding_strategy="infer_sample",
                start_tokens=[self.bos_id] * self.batch_size,
                end_token=self.eos_id,
                embedding=self.embedding_matrix,
                initial_state=initial_state)

    @staticmethod
    def default_hparams():
        return {
            'name': 'EmbeddingTiedLanguageModel',
            'embedding_dim': 400,
            'num_layers': 1,
            'hidden_units': 1150,
            'input_dropout': 0.6,
            'output_dropout': 0.7,
            'state_dropout': 0.55,
            'intra_layer_dropout': 0.4,
            'embedding_dropout': 0.0,
            'variational_recurrent': True,
        }
