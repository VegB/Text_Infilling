train_file = "../../data/coco.txt"
vocab_file = "../../data/coco_vocab.txt"
# train_file = "../../data/small_sent.txt"
# vocab_file = "../../data/vocab.txt"
positive_file = "./data/positive.txt"
negative_file = "./data/negative.txt"
log_file = "./data/log.txt"
ckpt = "./checkpoint/ckpt"

init_scale = 0.1
rnn_dim = 128
latent_num = 100
keep_prob = 0.7
batch_size = 20
num_steps = 20
print_num = 5

target_pretrain_epoch = 80
generator_pretrain_epoch = 80
discriminator_pretrain_epoch = 80
adversial_epoch = 100
g_update_batch = 1
d_update_batch = 1

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": rnn_dim,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 3
}
d_cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": rnn_dim,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": 1.0},
    "num_layers": 3
}
emb = {
    "dim": rnn_dim
}
g_opt = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.0001
        }
    }
}
d_opt = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.0001
        }
    }
}
reward_opt = {
    "optimizer": {
        "type": "GradientDescentOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    }
}
teacher_opt = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    }
}
