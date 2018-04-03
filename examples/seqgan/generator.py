import tensorflow as tf

from txtgen.modules import ForwardConnector
from txtgen.modules import BasicRNNDecoder, get_helper
from txtgen.modules import HierarchicalEncoder
from txtgen.losses import mle_losses
from txtgen.core import optimization as opt
from txtgen.core import layers

from utils import *


class Generator:
    def __init__(self, batch_size, max_seq_length, word2id, bos, eos, pad, emb_dim=None, word2vec=None):
        with tf.variable_scope('generator'):
            self.batch_size = batch_size
            self.max_seq_length = max_seq_length
            self.vocab_size = len(word2id)
            self.reward_gamma = 0.9
            self.embedding = create_word_embedding(word2id, emb_dim, word2vec)

            self.data_batch = tf.placeholder(dtype=tf.int32, name="data_batch",
                                             shape=(self.batch_size, self.max_seq_length + 2))
            self.rewards = tf.placeholder(dtype=tf.float32, name='rewards',
                                          shape=[self.batch_size, self.max_seq_length])
            self.expected_reward = tf.Variable(tf.zeros([self.max_seq_length]))
            self.bos_id = bos
            self.eos_id = eos
            self.pad_id = pad

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

            # Build decoder. Simply use the default hyperparameters.\
            self.decoder = BasicRNNDecoder(vocab_size=self.vocab_size)

            # Build connector
            self.connector = ForwardConnector(self.decoder.state_size)

            enc_outputs, enc_last = self.encoder(inputs=self.data_batch[:, tf.newaxis])  # [batch, 1, max_len + 2]

            helper_train = get_helper(
                self.decoder.hparams.helper_train.type,
                inputs=self.data_batch[:, :-1],   # [batch, max_len + 1]
                sequence_length=[self.max_seq_length + 1] * self.batch_size,
                embedding=self.decoder.embedding)

            # Decode for training
            self.outputs, final_state, sequence_lengths = self.decoder(
                helper=helper_train, initial_state=self.connector(enc_last))

            # Build loss
            self.mle_loss = mle_losses.average_sequence_sparse_softmax_cross_entropy(
                labels=self.data_batch[:, 1:],  # [batch, max_len + 1]
                logits=self.outputs.rnn_output,  # [batch, max_len + 1, vocab_size]
                sequence_length=sequence_lengths - 1)

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

            # build loss for updating with D predictions
            true_sample = self.data_batch[:, 1:self.max_seq_length + 1]  # [batch, max_len]
            g_predictions = self.outputs.rnn_output[:, :self.max_seq_length, :]  # [batch, max_len, vocab_size]

            self.update_loss = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(true_sample, [-1])), self.vocab_size, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(g_predictions, [-1, self.vocab_size]), 1e-20, 1.0)
                    ), 1) * tf.reshape(self.rewards, [-1])
            )
            self.update_op, self.update_step = opt.get_train_op(self.update_loss, hparams=opt_hparams)

            # for generation
            helper_infer = get_helper(
                self.decoder.hparams.helper_infer.type,
                embedding=self.decoder.embedding,
                start_tokens=[self.bos_id] * self.batch_size,
                end_token=self.eos_id,
                softmax_temperature=None,
                seed=None)

            self.generated_outputs, final_state, sequence_lengths = self.decoder(
                helper=helper_infer, initial_state=self.connector(enc_last))
