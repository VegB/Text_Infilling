# -*- coding: utf-8 -*-
"""
configurate the hyperparameters, based on command line arguments.
"""
import argparse
import math
import os

from texar.utils.transformer_utils import _batching_scheme
from texar.data import SpecialTokens


class Hyperparams:
    """
        config dictionrary, initialized as an empty object.
        The specific values are passed on with the ArgumentParser
    """
    def __init__(self):
        self.help = "the hyperparams dictionary to use"


def load_hyperparams():
    """
        main function to define hyperparams
    """
    # pylint: disable=too-many-statements
    args = Hyperparams()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_seq_length', type=int, default=16)  # 256
    argparser.add_argument('--running_mode', type=str,
                           default='train_and_evaluate',
                           help='can also be test mode')
    argparser.add_argument('--filename_prefix', type=str, default='yahoo.')
    argparser.add_argument('--debug', type=int, default=0)
    argparser.add_argument('--draw_for_debug', type=int, default=1)
    argparser.add_argument('--average_model', type=int, default=0,
                           help='currently not supported')
    argparser.add_argument('--model_dir', type=str, default='default')
    argparser.add_argument('--model_filename', type=str, default='')
    argparser.add_argument('--verbose', type=int, default=0)
    argparser.add_argument('--zero_pad', type=int, default=0)
    argparser.add_argument('--bos_pad', type=int, default=0,
                           help='use all-zero embedding for bos')
    argparser.add_argument('--data_dir', type=str,
                           default='./yahoo_data/')
    argparser.add_argument('--batch_size', type=int, default=64)  # 4096
    argparser.add_argument('--test_batch_size', type=int, default=10)
    argparser.add_argument('--min_length_bucket', type=int, default=9)
    argparser.add_argument('--length_bucket_step', type=float, default=1.1)
    argparser.add_argument('--max_training_steps', type=int, default=2500000)
    argparser.add_argument('--warmup_steps', type=int, default=16000)
    argparser.add_argument('--lr_constant', type=float, default=2)
    argparser.add_argument('--max_train_epoch', type=int, default=150)
    argparser.add_argument('--random_seed', type=int, default=1234)
    argparser.add_argument('--log_disk_dir', type=str, default='./')
    argparser.add_argument('--beam_width', type=int, default=2)
    argparser.add_argument('--alpha', type=float, default=0.6,
                           help=' length_penalty=(5+len(decode)/6) ^ -\alpha')
    argparser.add_argument('--save_eval_output', default=1,
                           help='save the eval output to file')
    argparser.add_argument('--eval_interval_epoch', type=int, default=1)
    argparser.add_argument('--load_from_pytorch', type=str, default='')
    argparser.add_argument('--affine_bias', type=int, default=0)
    argparser.add_argument('--eval_criteria', type=str, default='bleu')
    argparser.add_argument('--pre_encoding', type=str, default='spm')
    argparser.add_argument('--mask_strategy', type=str, default='random')  # equal_length
    argparser.add_argument('--present_rate', type=float, default=0.2)
    argparser.add_argument('--mask_num', type=int, default=3)
    argparser.add_argument('--mask_length', type=int, default=5)
    argparser.add_argument('--hidden_dim', type=int, default=512)
    argparser.parse_args(namespace=args)

    args.max_decode_len = args.max_seq_length
    args.max_partition_num = int((args.max_seq_length + 1) / 2)
    args.partition_num = int(math.log(args.max_seq_length))
    args.data_dir = os.path.abspath(args.data_dir)
    args.filename_suffix = '.txt'
    args.train_file = os.path.join(args.data_dir,
        '{}train{}'.format(args.filename_prefix, args.filename_suffix))
    args.valid_file = os.path.join(args.data_dir,
        '{}valid{}'.format(args.filename_prefix, args.filename_suffix))
    args.test_file = os.path.join(args.data_dir,
        '{}test{}'.format(args.filename_prefix, args.filename_suffix))
    args.vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    if args.mask_strategy == 'random':
        log_params_dir = 'log_dir/{}bsize{}.epoch{}.seqlen{}.{}.present{}.partition{}.hidden{}.seq2seq/'.format(
            args.filename_prefix, args.batch_size, args.max_train_epoch, args.max_seq_length,
            args.mask_strategy, args.present_rate, args.partition_num, args.hidden_dim)
    elif args.mask_strategy == 'equal_length':
        log_params_dir = 'log_dir/{}bsize{}.epoch{}.seqlen{}.{}.masknum{}.masklen{}.hidden{}.seq2seq/'.format(
            args.filename_prefix, args.batch_size, args.max_train_epoch, args.max_seq_length,
            args.mask_strategy, args.mask_num, args.mask_len, args.hidden_dim)
    args.log_dir = os.path.join(args.log_disk_dir, log_params_dir)
    batching_scheme = _batching_scheme(
        args.batch_size,
        args.max_seq_length,
        args.min_length_bucket,
        args.length_bucket_step,
        drop_long_sequences=True,
    )
    print('train_file:{}'.format(args.train_file))
    print('valid_file:{}'.format(args.valid_file))
    train_dataset_hparams = {
        "num_epochs": args.eval_interval_epoch,
        "seed": args.random_seed,
        "shuffle": True,
        "dataset": {
            "files": args.train_file,
            "vocab_file": args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        # 'bucket_boundaries': batching_scheme['boundaries'],
        # 'bucket_batch_sizes': batching_scheme['batch_sizes'],
        'batch_size': args.batch_size,
        'allow_smaller_final_batch': True,
    }
    eval_dataset_hparams = {
        "num_epochs": 1,
        'seed': args.random_seed,
        'shuffle': False,
        'dataset': {
            'files': args.valid_file,
            'vocab_file': args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        'batch_size': args.test_batch_size,
        'allow_smaller_final_batch': True,
    }
    test_dataset_hparams = {
        "num_epochs": 1,
        "seed": args.random_seed,
        "shuffle": False,
        "dataset": {
            "files": args.test_file,
            "vocab_file": args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        'batch_size': args.test_batch_size,
        'allow_smaller_final_batch': True,
    }
    args.word_embedding_hparams = {
        'name': 'lookup_table',
        'dim': args.hidden_dim,
        'initializer': {
            'type': 'random_normal_initializer',
            'kwargs': {
                'mean': 0.0,
                'stddev': args.hidden_dim**-0.5,
            },
        }
    }
    cell = {
        "type": "LSTMBlockCell",
        "kwargs": {
            "num_units": args.hidden_dim*4,
            "forget_bias": 0.
        },
        "dropout": {"output_keep_prob": 1-0.1},
        "num_layers": 1
    }
    output_layer = {
        "num_layers": 0,
        "layer_size": args.hidden_dim*4,
        "activation": "identity",
        "final_layer_activation": None,
        "other_dense_kwargs": None,
        "dropout_layer_ids": [],
        "dropout_rate": 0.1,
        "variational_dropout": False,
        "@no_typecheck": ["activation", "final_layer_activation",
                          "layer_size", "dropout_layer_ids"]
    }
    encoder_hparams = {
        "rnn_cell": cell,
        "output_layer": output_layer,
        "name": "unidirectional_rnn_encoder"
    }
    decoder_hparams = {
        "rnn_cell": cell,
        "max_decoding_length_train": args.max_seq_length+2,
        "max_decoding_length_infer": args.max_seq_length+2,
        "name": "basic_rnn_decoder"
    }
    loss_hparams = {
        'label_confidence': 0.9,
    }

    opt_hparams = {
        'learning_rate_schedule':
            'constant.linear_warmup.rsqrt_decay.rsqrt_depth',
        'lr_constant': args.lr_constant,
        'warmup_steps': args.warmup_steps,
        'max_training_steps': args.max_training_steps,
        'Adam_beta1': 0.9,
        'Adam_beta2': 0.997,
        'Adam_epsilon': 1e-9,
    }
    print('logdir:{}'.format(args.log_dir))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + 'img/'):
        os.makedirs(args.log_dir + 'img/')
    return {
        'train_dataset_hparams': train_dataset_hparams,
        'eval_dataset_hparams': eval_dataset_hparams,
        'test_dataset_hparams': test_dataset_hparams,
        'encoder_hparams': encoder_hparams,
        'decoder_hparams': decoder_hparams,
        'loss_hparams': loss_hparams,
        'opt_hparams': opt_hparams,
        'args': args,
        }
