from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import texar as tx


def embedding_drop(embedding_matrix, keep_prob):
    mask = tf.nn.dropout(tf.ones((embedding_matrix.shape[0], 1)), keep_prob)
    return mask * embedding_matrix


class Generator(tx.modules.ModuleBase):
    def __init__(self, vocab_size, hparams=None):
        tx.ModuleBase.__init__(self, hparams)
        self.vocab_size = vocab_size
        self.embedding_dim = self.hparams.embedding_dim
        self.num_layers = self.hparams.num_layers
        self.hidden_units = self.hparams.hidden_units
        self.input_dropout = self.hparams.input_dropout
        self.output_dropout = self.hparams.output_dropout
        self.state_dropout = self.hparams.state_dropout
        self.intra_layer_dropout = self.hparams.intra_layer_dropout
        self.embedding_dropout = self.hparams.embedding_dropout
        self.variational_recurrent = self.hparams.variational_recurrent

        with tf.variable_scope('generator'):
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
                cell=cell, vocab_size=self.embedding_dim)


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

    def _build(self, text_ids, num_steps):
        embedding_matrix = tf.transpose(self.output_layer.weights[0])
        self.embedding_matrix = embedding_drop(
            embedding_matrix,
            tx.utils.switch_dropout(1. - self.embedding_dropout))

        initial_state = self.decoder.zero_state(
            batch_size=tf.shape(text_ids)[0], dtype=tf.float32)
        outputs, final_state, sequence_length = self.decoder(
            inputs=tf.nn.embedding_lookup(self.embedding_matrix, text_ids),
            initial_state=initial_state,
            impute_finished=True,
            decoding_strategy="train_greedy",
            sequence_length=num_steps)
        logits = self.output_layer(outputs.logits)
        sample_id = tf.argmax(logits, 2)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return initial_state, logits, final_state, sample_id
