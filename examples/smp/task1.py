"""
Classifier for Task 1.
"""

import importlib
import tensorflow as tf
import texar as tx


flags = tf.flags
flags.DEFINE_string("data_path", "./toy", "Directory containing data.")
flags.DEFINE_string("config", "config", "The config to use.")
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)

train_data = tx.data.MultiAlignedData(config.train_data_hparams)
iterator = tx.data.DataIterator(train_data)
data_batch = iterator.get_next()

