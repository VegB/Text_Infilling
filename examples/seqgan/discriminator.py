import tensorflow as tf

import texar as tx

from utils import *


class Discriminator:
    def __init__(self, config, word2id, class_num=2):
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)
        with tf.variable_scope('discriminator', initializer=initializer):
            self.batch_size = config.batch_size
            self.max_seq_length = config.num_steps
            self.vocab_size = len(word2id)
            self.class_num = class_num

            self.labels = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, 1), name="labels")
            self.samples = tf.placeholder(dtype=tf.int32, name="samples",
                                          shape=(self.batch_size, self.max_seq_len + 1))

            self.embedder = tx.modules.WordEmbedder(
                vocab_size=self.vocab_size, hparams=config.emb)

            # build Encoder
            hparams = {
                "use_embedding": True,
                "embedding": layers.default_embedding_hparams(),
                "minor_type": "rnn",
                "minor_cell": layers.default_rnn_cell_hparams(),
                "major_cell": layers.default_rnn_cell_hparams(),
            }
            hparams["name"] = "hierarchical_forward_rnn_encoder"
            hparams["embedding"]["dim"] = emb_dim

            # builder encoder
            self.encoder = HierarchicalEncoder(vocab_size=self.vocab_size,
                                               embedding=self.embedding, hparams=hparams)
            self.encoder_unit_num = self.encoder.hparams.minor_cell.cell.kwargs.num_units

            enc_outputs, enc_last = self.encoder(inputs=self.samples[:, tf.newaxis])

            # build classifying layer params
            self.W = tf.Variable(tf.random_uniform([self.encoder_unit_num, class_num], -1.0, 1.0), name="W")
            self.b = tf.Variable(tf.constant(0.1, shape=[class_num]), name="b")

            # Make predictions for training
            hidden_state = enc_last[1]
            self.scores = tf.nn.xw_plus_b(hidden_state, self.W, self.b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.ypred_for_auc, 1, name="predictions")

            # Calculate loss
            self.mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
                labels=self.labels,  # [batch, 1]
                logits=self.scores[:, tf.newaxis],  # [batch, 1, num_class]
                sequence_length=[1] * self.batch_size)

            # Build train op. Only config the optimizer while using default settings
            # for other hyperparameters.
            opt_hparams = {
                "optimizer": {
                    "type": "MomentumOptimizer",
                    "kwargs": {
                        "learning_rate": 0.01,
                        "momentum": 0.9
                    }
                }
            }
            self.train_op, self.global_step = opt.get_train_op(self.mle_loss, hparams=opt_hparams)
