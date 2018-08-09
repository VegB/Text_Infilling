generator_pretrain_epoch = 80
discriminator_pretrain_epoch = 80
adversial_epoch = 100
hidden_size = 32
batch_size = 64
max_num_steps = 20
embed_dim = 32
latent_dims = 32

enc_keep_prob_in = 1.0
dec_keep_prob_out = 1.0

log_dir = './log_dir/'
log_file = log_dir + 'log.txt'
bleu_file = log_dir + 'bleu.txt'
ckpt = './checkpoint/ckpt'

lr_hparams = {
    'init_lr': 0.003,
    'update_init_lr': 0.0003,
    'update_lr': 0.00003,
    'decay_rate': 0.1,
    'threshold': 5
}

decoder_hparams = {
    "type": "lstm"
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
    'initializer': {
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
        "files": 'coco_data/coco.train.txt',
        "vocab_file": 'coco_data/vocab.txt'
    }
}

val_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
        "files": 'coco_data/coco.valid.txt',
        "vocab_file": 'coco_data/vocab.txt'
    }
}

test_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "dataset": {
        "files": 'coco_data/coco.test.txt',
        "vocab_file": 'coco_data/vocab.txt'
    }
}

g_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.01
        }
    }
}

d_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.0001
        }
    }
}

update_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.0004
        }
    }
}
