generator_pretrain_epoch = 120
discriminator_pretrain_epoch = 80
adversial_epoch = 100
hidden_size = 256
batch_size = 64
max_num_steps = 64
embed_dim = 256
latent_dims = 32

dec_keep_prob_in = 0.5
dec_keep_prob_out = 0.5
enc_keep_prob_in = 1.0
enc_keep_prob_out = 1.0

log_file = './log.txt'
ckpt = './checkpoint/ckpt'

lr_hparams = {
    'init_lr': 0.003,
    'update_init_lr': 0.0003,
    'update_lr': 0.00003,
    'decay_rate': 0.1
}

decoder_hparams = {
    "type": "lstm"
}

enc_cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": enc_keep_prob_out},
    "num_layers": 1
}

dec_cell_hparams = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": dec_keep_prob_out},
    "num_layers": 1
}

emb_hparams = {
    'name': 'lookup_table',
    "dim": embed_dim,
    'initializer' : {
        'type': 'random_normal_initializer',
        'kwargs': {
            'mean': 0.0,
            'stddev': embed_dim**-0.5,
        },
    }
}


# KL annealing
kl_anneal_hparams={
    "warm_up": 10,
    "start": 0.01
}

train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'ptb_data/ptb.train.txt',
        "vocab_file": 'ptb_data/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'ptb_data/ptb.valid.txt',
        "vocab_file": 'ptb_data/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": 'ptb_data/ptb.test.txt',
        "vocab_file": 'ptb_data/vocab.txt'
    }
}

d_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    }
}

update_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    }
}
