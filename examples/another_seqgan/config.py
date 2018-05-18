# train_file = "../../data/coco.txt"
# coco_file = "../../data/coco.txt"
# valid_file = "../../data/coco.txt"
# vocab_file = "../../data/coco_vocab.txt"
# '''
# train_file = "../../data/ptb_train.txt"
# test_file = "../../data/ptb_test.txt"
# valid_file = "../../data/ptb_valid.txt"
# vocab_file = "../../data/ptb_vocab.txt"
train_file = "./data/ptb.train.txt"
test_file = "./data/ptb.test.txt"
valid_file = "./data/ptb.valid.txt"
'''
train_file = "../../data/small_sent.txt"
test_file = "../../data/small_sent.txt"
valid_file = "../../data/small_sent.txt"
vocab_file = "../../data/vocab.txt"
'''
positive_file = "./data/positive.txt"
negative_file = "./data/negative.txt"
log_file = "./data/log.txt"
train_log_file = "./data/training_log.txt"
eval_log_file = "./data/eval_log.txt"
ckpt = "./checkpoint/ckpt"

init_scale = 0.1
rnn_dim = 400
rnn_layers = 1
keep_prob = 1.0
batch_size = 64
num_steps = 35
print_num = 3
init_lr = 0.003
min_lr = 0.0000001
l2_decay = 1e-5
lr_decay = 0.1

target_pretrain_epoch = 80
generator_pretrain_epoch = 130
discriminator_pretrain_epoch = 80
adversial_epoch = 100
g_update_batch = 1
d_update_batch = 1

d_cell = {
    "type": "LSTMBlockCell",
    "kwargs": {
        "num_units": rnn_dim,
        "forget_bias": 0.
    },
    "dropout": {"output_keep_prob": 1.0},
    "num_layers": rnn_layers
}
emb = {
    "dim": rnn_dim
}
opt = {
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
