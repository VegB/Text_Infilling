# -*- coding: utf-8 -*-
"""
configurate the hyperparameters, based on command line arguments.
"""
import argparse
import copy
import os

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
    argparser.add_argument('--blank_num', type=int, default=3)
    argparser.add_argument('--batch_size', type=int, default=150)  # 4096
    argparser.add_argument('--max_seq_length', type=int, default=20)  #256
    argparser.add_argument('--hidden_dim', type=int, default=512)
    argparser.add_argument('--running_mode', type=str,
                           default='train_and_evaluate',
                           help='can also be test mode')
    argparser.add_argument('--max_training_steps', type=int, default=2500000)
    argparser.add_argument('--warmup_steps', type=int, default=16000)
    argparser.add_argument('--max_train_epoch', type=int, default=350)
    argparser.add_argument('--bleu_interval', type=int, default=10)
    argparser.add_argument('--decay_interval', type=float, default=20)
    argparser.add_argument('--log_disk_dir', type=str, default='./')
    argparser.add_argument('--filename_prefix', type=str, default='yahoo.')
    argparser.add_argument('--data_dir', type=str,
                           default='./yahoo_data/')
    argparser.add_argument('--save_eval_output', default=1,
        help='save the eval output to file')
    argparser.add_argument('--lr_constant', type=float, default=0.3)
    argparser.add_argument('--lr_decay_rate', type=float, default=0.1)
    argparser.add_argument('--lr_factor', type=float, default=0.1)
    argparser.add_argument('--learning_rate_strategy', type=str, default='dynamic')  # 'static'
    argparser.add_argument('--zero_pad', type=int, default=0)
    argparser.add_argument('--bos_pad', type=int, default=0,
                           help='use all-zero embedding for bos')
    argparser.add_argument('--random_seed', type=int, default=1234)
    argparser.add_argument('--beam_width', type=int, default=2)
    argparser.add_argument('--affine_bias', type=int, default=0)
    argparser.parse_args(namespace=args)
    
    args.max_decode_len = args.max_seq_length
    args.data_dir = os.path.abspath(args.data_dir)
    args.filename_suffix = '.txt'
    args.vocab_file = os.path.join(args.data_dir, 'vocab.txt')
    log_params_dir = 'log_dir/{}bsize{}.epoch{}.seqlen{}.{}_lr.partition{}.hidden{}.self_attn/'.format(
        args.filename_prefix, args.batch_size, args.max_train_epoch, args.max_seq_length,
        args.learning_rate_strategy, args.blank_num, args.hidden_dim)
    args.log_dir = os.path.join(args.log_disk_dir, log_params_dir)

    data_files = {
        mode: {
            data_name: args.data_dir + '/' + args.filename_prefix + '%s.%s.txt' % (data_name, mode)
            for data_name in ['source', 'templatebyword', 'answer', 'start', 'end']
        }
        for mode in ['train', 'test', 'valid']
    }
    data_hparams = {
        stage: {
            "num_epochs": 1,
            "shuffle": stage != 'test',
            "batch_size": args.batch_size,
            "datasets": [
                {  # source
                    "files": [data_files[stage]['source']],
                    "vocab_file": os.path.join(args.vocab_file),
                    "max_seq_length": args.max_seq_length,
                    "bos_token": SpecialTokens.BOS,
                    "eos_token": SpecialTokens.EOS,
                    "length_filter_mode": "truncate",
                    "data_name": "source"
                },
                {  # templatebyword
                    "files": [data_files[stage]['templatebyword']],
                    "vocab_share_with": 0,
                    "max_seq_length": args.max_seq_length,
                    "data_name": "templatebyword"
                },
                {  # answer
                    "files": [data_files[stage]['answer']],
                    "vocab_share_with": 0,
                    "max_seq_length": args.max_seq_length,
                    "bos_token": SpecialTokens.BOA,
                    "eos_token": SpecialTokens.EOA,
                    "variable_utterance": True,
                    "data_name": "answer"
                }
            ]
        }
        for stage in ['train', 'valid', 'test']
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
        'num_blocks': 6,
        'num_heads': 8,
        'num_units': args.hidden_dim,
        'zero_pad': args.zero_pad,
        'bos_pad': args.bos_pad,
        'initializer': {
            'type': 'variance_scaling_initializer',
            'kwargs': {
                'scale': 1.0,
                'mode': 'fan_avg',
                'distribution':'uniform',
            },
        },
        'poswise_feedforward': {
            'name': 'ffn',
            'layers': [
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'conv1',
                        'units': args.hidden_dim*4,
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
                    'type':'Dense',
                    'kwargs': {
                        'name': 'conv2',
                        'units': args.hidden_dim,
                        'use_bias': True,
                        }
                }
            ],
        },
    }
    decoder_hparams = copy.deepcopy(encoder_hparams)
    decoder_hparams['share_embed_and_transform'] = True
    decoder_hparams['transform_with_bias'] = args.affine_bias
    decoder_hparams['maximum_decode_length'] = args.max_decode_len
    decoder_hparams['beam_width'] = args.beam_width
    decoder_hparams['sampling_method'] = 'argmax'
    loss_hparams = {
        'label_confidence': 0.9,
    }

    opt_hparams = {
        'learning_rate_schedule': args.learning_rate_strategy,
        'lr_constant': args.lr_constant,
        'warmup_steps': args.warmup_steps,
        'max_training_steps': args.max_training_steps,
        'Adam_beta1': 0.9,
        'Adam_beta2': 0.997,
        'Adam_epsilon': 1e-9,
    }
    opt_vars = {
        'learning_rate': args.hidden_dim ** -0.5 * 0.2 * args.lr_factor,  # 0.016
        'best_train_loss': 1e100,
        'best_eval_loss': 1e100,
        'best_eval_bleu': 0,
        'steps_not_improved': 0,
        'epochs_not_improved': 0,
        'decay_interval': args.decay_interval,
        'lr_decay_rate': args.lr_decay_rate,
        'decay_time': 0
    }
    print('logdir:{}'.format(args.log_dir))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + 'img/'):
        os.makedirs(args.log_dir + 'img/')
    return {
        'train_dataset_hparams': data_hparams['train'],
        'eval_dataset_hparams': data_hparams['valid'],
        'test_dataset_hparams': data_hparams['test'],
        'encoder_hparams': encoder_hparams,
        'decoder_hparams': decoder_hparams,
        'loss_hparams': loss_hparams,
        'opt_hparams': opt_hparams,
        'opt_vars': opt_vars,
        'args': args,
        }

