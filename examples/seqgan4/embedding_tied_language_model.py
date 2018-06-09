from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import texar as tx


def embedding_drop(embedding_matrix, keep_prob):
    mask = tf.nn.dropout(tf.ones((embedding_matrix.shape[0], 1)), keep_prob)
    return mask * embedding_matrix


class EmbeddingTiedLanguageModel(tx.modules.ModuleBase):
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

        with tf.variable_scope(self.variable_scope):
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

    def _build(self, text_ids, num_steps, infer=False, end_token=None):
        embedding_matrix = tf.transpose(self.output_layer.weights[0])
        embedding_matrix = embedding_drop(
            embedding_matrix,
            tx.utils.switch_dropout(1. - self.embedding_dropout))

        initial_state = self.decoder.zero_state(
            batch_size=tf.shape(text_ids)[0], dtype=tf.float32)
        if not infer:
            outputs, final_state, _ = self.decoder(
                inputs=tf.nn.embedding_lookup(embedding_matrix, text_ids),
                initial_state=initial_state,
                impute_finished=True,
                decoding_strategy="train_greedy",
                sequence_length=num_steps)
            rtn = (initial_state, self.output_layer(outputs.logits), final_state)
        else:
            infer_outputs, _, sequence_length = self.decoder(
                decoding_strategy="infer_sample",
                start_tokens=text_ids[:, 0],
                end_token=end_token,
                embedding=embedding_matrix,
                initial_state=initial_state,
                max_decoding_length=num_steps[0])
            infer_logits = self.output_layer(infer_outputs.logits)
            infer_sample_id = tf.argmax(infer_logits, 2)
            rtn = (infer_logits, infer_sample_id, sequence_length)

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return rtn
