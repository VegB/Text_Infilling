train_file = "../../data/oracle.txt"
vocab_file = "../../data/oracle_vocab.txt"
# train_file = "../../data/small_sent.txt"
# vocab_file = "../../data/vocab.txt"
positive_file = "./data/positive.txt"
negative_file = "./data/negative.txt"
log_file = "./data/log.txt"
ckpt = "./checkpoint/"

init_scale = 0.1
num_epochs = 13
hidden_size = 32
keep_prob = 1.0
batch_size = 20
num_steps = 20

rollout_num = 16
target_pretrain_epoch = 80
generator_pretrain_epoch = 80
discriminator_pretrain_epoch = 80
adversial_epoch = 100
adv_g_step = 1
adv_d_epoch = 15

cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": hidden_size,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": keep_prob},
    "num_layers": 2
}
emb = {
    "dim": hidden_size
}
opt = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.01
        }
    }
}
cnn = {
      "kernel_sizes": [2, 3],
      # "num_filter": [100, 200]
    }
