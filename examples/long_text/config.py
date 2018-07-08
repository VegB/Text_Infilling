# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=invalid-name, too-few-public-methods, missing-docstring

import copy

num_epochs = 50
hidden_size = 512
enc_keep_prob_in = 1.0
enc_keep_prob_out = 1.0
dec_keep_prob_in = 0.5
batch_size = 32
embed_dim = 512
latent_dims = 32

beam_width = 5
max_decode_len = 256
max_seq_length = 256

is_present_rate = 0.5

lr_decay_hparams = {
    "init_lr": 0.001,
    "threshold": 1,
    "rate": 0.1
}

relu_dropout = 0.2
embedding_dropout = 0.2
attention_dropout = 0.2
residual_dropout = 0.2
num_blocks = 3

emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    'initializer': {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': embed_dim**-0.5,
        },
    }
}

encoder_hparams = {
    'multiply_embedding_mode': "sqrt_depth",
    'embedding_dropout': 0.1,
    'position_embedder': {
        'name': 'sinusoids',
        'hparams': None,
    },
    'attention_dropout': 0.1,
    'residual_dropout': 0.1,
    'sinusoid': True,
    'num_blocks': num_blocks,
    'num_heads': 8,
    'num_units': hidden_size,
    'zero_pad': False,
    'bos_pad': False,
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': {
        'name': 'ffn',
        'layers': [{
                'type': 'Dense',
                'kwargs': {
                    'name': 'conv1',
                    'units': hidden_size*4,
                    'activation': 'relu',
                    'use_bias': True,
                }
            },
            {
                'type': 'Dropout',
                'kwargs': {
                    'rate': 0.1,
                }
            },
            {
                'type': 'Dense',
                'kwargs': {
                    'name': 'conv2',
                    'units': hidden_size,
                    'use_bias': True,
                    }
            }
        ],
    },
}

decoder_hparams = copy.deepcopy(encoder_hparams)
decoder_hparams['share_embed_and_transform'] = True
decoder_hparams['transform_with_bias'] = False
decoder_hparams['maximum_decode_length'] = max_decode_len
decoder_hparams['beam_width'] = beam_width
decoder_hparams['sampling_method'] = 'argmax'

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'yahoo_data/yahoo.train.txt',
        "vocab_file": 'yahoo_data/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'yahoo_data/yahoo.valid.txt',
        "vocab_file": 'yahoo_data/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": 'yahoo_data/yahoo.test.txt',
        "vocab_file": 'yahoo_data/vocab.txt'
    }
}
