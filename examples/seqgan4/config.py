data_hparams = {
    "train": "../another_seqgan/data/ptb.train.txt",
    "test": "../another_seqgan/data/ptb.test.txt",
    "valid": "../another_seqgan/data/ptb.valid.txt"
}
lr_hparams = {
    'init_lr': 0.003,
    'update_init_lr': 0.0003,
    'update_lr': 0.00003
}
l2_hparams = {
    'l2_decay': 1e-5,
    'lr_decay': 0.1
}
opt_vars = {
    'learning_rate': lr_hparams['init_lr'],
    'update_learning_rate': lr_hparams['update_lr'],
    'best_valid_ppl': 1e100,
    'steps_not_improved': 0
}
training_hparams = {
    'batch_size': 64,
    'num_steps': 35,
    'generator_pretrain_epoch': 120,
    'discriminator_pretrain_epoch': 80,
    'adversial_epoch': 100
}
log_hparams = {
    'log_file': "./data/log.txt",
    'train_log_file': "./data/training_log.txt",
    'eval_log_file': "./data/eval_log.txt"
}
emb_hparams = {
    "dim": 400
}
opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    }
}
