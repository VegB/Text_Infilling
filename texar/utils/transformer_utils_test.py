import math
import argparse
import numpy as np
import tensorflow as tf
from texar.utils.transformer_utils import prepare_template, _split_template, \
    _merge_segments, fill_template, update_template_pack


class Hyperparams:
    """
        config dictionrary, initialized as an empty object.
        The specific values are passed on with the ArgumentParser
    """
    def __init__(self):
        self.help = "the hyperparams dictionary to use"


def load_hyperparams():
    args = Hyperparams()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_seq_length', type=int, default=11)
    argparser.add_argument('--max_decode_len', type=int, default=5)
    argparser.add_argument('--mask_strategy', type=str, default='random')  # equal_length
    argparser.add_argument('--present_rate', type=float, default=0.5)
    argparser.add_argument('--mask_num', type=int, default=3)
    argparser.add_argument('--mask_length', type=int, default=5)
    argparser.parse_args(namespace=args)
    args.partition_num = 3
    return args


def test_prepare_template():
    data_batch = {
        'source_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'off', b'his', b'shoulder', b'and', b'up', b'the', b'steps', b'before', b'him', b'<EOS>'],
                                 [b'<BOS>', b'and', b'they', b'gave', b'hans', b'gifts', b'of', b'gold', b'and', b'of', b'silver', b'<EOS>', b'', b'']], dtype=object),
        'source_length': tf.Variable([14, 12], dtype=tf.int32),
        'source_text_ids': tf.Variable([[1,   10,   47, 1068,   44,  166, 1990,   10,  287,   49, 1401, 143,  115,    2],
                                     [1,   10,   19,   48, 1913,  775,  106,  778,   10,  106,  477, 2,    0,    0]]),
        'templatebyword_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'<m>', b'his', b'shoulder', b'and', b'<m>', b'<m>', b'steps', b'<m>', b'him', b'<EOS>'],
                                         [b'<BOS>', b'and', b'they', b'gave', b'<m>', b'hans', b'gifts', b'<m>', b'gold', b'and', b'<m>', b'silver', b'<EOS>', b'']], dtype=object),
        'templatebyword_length': tf.Variable([14, 13], dtype=tf.int32),
        'templatebyword_text_ids': tf.Variable([[1,   10,   47, 1068,    6,  166, 1990,   10,    6,    6, 1401, 6,  115,    2],
                                          [1,   10,   19,   48,    6, 1913,  775,    6,  778,   10,    6, 477,    2,    0]]),
        'answer_text': tf.Variable([[[b'<BOA>', b'off', b'<EOA>', b'<PAD>'],
                                  [b'<BOA>', b'up', b'the', b'<EOA>'],
                                  [b'<BOA>', b'before', b'<EOA>', b'<PAD>']],
                                 [[b'<BOA>', b'<EOA>', b'<PAD>', b''],
                                  [b'<BOA>', b'of', b'<EOA>', b''],
                                  [b'<BOA>', b'of', b'<EOA>', b'']]], dtype=object),
        'answer_length': tf.Variable([[3, 4, 3], [2, 3, 3]], dtype=tf.int32),
        'answer_text_ids': tf.Variable([[[4,  44,   5,   0],
                                    [4, 287,  49,   5],
                                    [4, 143,   5,   0]],
                                   [[4,   5,   0,   0],
                                    [4, 106,   5,   0],
                                    [4, 106,   5,   0]]]),
        'answer_utterance_cnt': tf.Variable([3, 3], dtype=tf.int32)
    }
    mask_id = 6
    pad_id = 0
    args = load_hyperparams()
    template_pack, answer_packs = \
        prepare_template(data_batch, args, mask_id, pad_id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fetches = {
            'ori': data_batch,
            'template': template_pack,
            'fills': answer_packs
        }
        rtns = sess.run(fetches)
        print(rtns['ori'])
        print(rtns['template'])
        print(rtns['fills'])
# test_prepare_template()


def test_split_template():
    a = [3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1]
    s_pos = [3, 8]
    e_pos = [5, 10]
    assert _split_template(a, s_pos, e_pos) == [[3, 5, 4], [1, 3, 3], [1]]
# test_split_template()


def test_merge_segments():
    t_seg_1 = [[3, 5, 4], [1, 3, 3], [1]]
    fillings_1 = [[4, 2], [2, 5]]
    assert _merge_segments(t_seg_1, fillings_1) == \
           [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    assert _merge_segments(t_seg_1[:-1], fillings_1) == \
           [3, 5, 4, 4, 2, 1, 3, 3, 2, 5]


def test_fill_template():
    templates = np.tf.Variable([[3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1],
                          [2, 1, 7, 7, 6, 2, 5, 7, 7, 4, 5]])
    predictions = np.tf.Variable([[[4, 2], [2, 5]], [[4, 3], [3, 1]]])
    mask_id = 7
    rst = fill_template(templates, predictions, mask_id)
    assert rst == [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                   [2, 1, 4, 3, 6, 2, 5, 3, 1, 4, 5]]


def test_fill_template_with_tensor():
    data_batch = {
        'source_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'off', b'his', b'shoulder', b'and', b'up',
                                     b'the', b'steps', b'before', b'him', b'<EOS>'],
                                    [b'<BOS>', b'and', b'they', b'gave', b'hans', b'gifts', b'of', b'gold', b'and',
                                     b'of', b'silver', b'<EOS>', b'', b'']], dtype=object),
        'source_length': tf.Variable([14, 12], dtype=tf.int32),
        'source_text_ids': tf.Variable([[1, 10, 47, 1068, 44, 166, 1990, 10, 287, 49, 1401, 143, 115, 2],
                                        [1, 10, 19, 48, 1913, 775, 106, 778, 10, 106, 477, 2, 0, 0]]),
        'templatebyword_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'<m>', b'his', b'shoulder', b'and',
                                             b'<m>', b'<m>', b'steps', b'<m>', b'him', b'<EOS>'],
                                            [b'<BOS>', b'and', b'they', b'gave', b'<m>', b'hans', b'gifts', b'<m>',
                                             b'gold', b'and', b'<m>', b'silver', b'<EOS>', b'']], dtype=object),
        'templatebyword_length': tf.Variable([14, 13], dtype=tf.int32),
        'templatebyword_text_ids': tf.Variable([[1, 10, 47, 1068, 6, 166, 1990, 10, 6, 6, 1401, 6, 115, 2],
                                                [1, 10, 19, 48, 6, 1913, 775, 6, 778, 10, 6, 477, 2, 0]]),
        'answer_text': tf.Variable([[[b'<BOA>', b'off', b'<EOA>', b'<PAD>'],
                                     [b'<BOA>', b'up', b'the', b'<EOA>'],
                                     [b'<BOA>', b'before', b'<EOA>', b'<PAD>']],
                                    [[b'<BOA>', b'<EOA>', b'<PAD>', b''],
                                     [b'<BOA>', b'of', b'<EOA>', b''],
                                     [b'<BOA>', b'of', b'<EOA>', b'']]], dtype=object),
        'answer_length': tf.Variable([[3, 4, 3], [2, 3, 3]], dtype=tf.int32),
        'answer_text_ids': tf.Variable([[[4, 44, 5, 0],
                                         [4, 287, 49, 5],
                                         [4, 143, 5, 0]],
                                        [[4, 5, 0, 0],
                                         [4, 106, 5, 0],
                                         [4, 106, 5, 0]]]),
        'answer_utterance_cnt': tf.Variable([3, 3], dtype=tf.int32)
    }
    args = load_hyperparams()
    mask_id = 6
    boa_id = 4
    eoa_id = 5
    eos_id = 2
    pad_id = 0
    template_pack, answer_packs = prepare_template(data_batch, args, mask_id, pad_id)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fetches = {
            'ori': data_batch,
            'template': template_pack,
            'fills': answer_packs
        }
        rtns = sess.run(fetches)
        print(rtns['ori'])
        print(rtns['template'])
        print(rtns['fills'])
        predictions = []
        for hole in rtns['fills']:
            predictions.append(hole['text_ids'][:, 1:])

        filled = fill_template(template_pack=rtns['template'],
                               predictions=predictions,
                               eoa_id=eoa_id, pad_id=pad_id, eos_id=eos_id)
        print("\noriginal:\n", rtns['ori']['source_text_ids'].tolist())
        print("\ntemplate:\n", rtns['template']['text_ids'].tolist())
        print("\nfilled:\n", filled)
        assert filled == rtns['ori']['source_text_ids'].tolist()
# test_fill_template_with_tensor()


def test_update_template_pack():
    data_batch = {
        'source_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'off', b'his', b'shoulder', b'and', b'up',
                                     b'the', b'steps', b'before', b'him', b'<EOS>'],
                                    [b'<BOS>', b'and', b'they', b'gave', b'hans', b'gifts', b'of', b'gold', b'and',
                                     b'of', b'silver', b'<EOS>', b'', b'']], dtype=object),
        'source_length': tf.Variable([14, 12], dtype=tf.int32),
        'source_text_ids': tf.Variable([[1, 10, 47, 1068, 44, 166, 1990, 10, 287, 49, 1401, 143, 115, 2],
                                        [1, 10, 19, 48, 1913, 775, 106, 778, 10, 106, 477, 2, 0, 0]]),
        'templatebyword_text': tf.Variable([[b'<BOS>', b'and', b'she', b'sprang', b'<m>', b'his', b'shoulder', b'and',
                                             b'<m>', b'<m>', b'steps', b'<m>', b'him', b'<EOS>'],
                                            [b'<BOS>', b'and', b'they', b'gave', b'<m>', b'hans', b'gifts', b'<m>',
                                             b'gold', b'and', b'<m>', b'silver', b'<EOS>', b'']], dtype=object),
        'templatebyword_length': tf.Variable([14, 13], dtype=tf.int32),
        'templatebyword_text_ids': tf.Variable([[1, 10, 47, 1068, 6, 166, 1990, 10, 6, 6, 1401, 6, 115, 2],
                                                [1, 10, 19, 48, 6, 1913, 775, 6, 778, 10, 6, 477, 2, 0]]),
        'answer_text': tf.Variable([[[b'<BOA>', b'off', b'<EOA>', b'<PAD>'],
                                     [b'<BOA>', b'up', b'the', b'<EOA>'],
                                     [b'<BOA>', b'before', b'<EOA>', b'<PAD>']],
                                    [[b'<BOA>', b'<EOA>', b'<PAD>', b''],
                                     [b'<BOA>', b'of', b'<EOA>', b''],
                                     [b'<BOA>', b'of', b'<EOA>', b'']]], dtype=object),
        'answer_length': tf.Variable([[3, 4, 3], [2, 3, 3]], dtype=tf.int32),
        'answer_text_ids': tf.Variable([[[4, 44, 5, 0],
                                         [4, 287, 49, 5],
                                         [4, 143, 5, 0]],
                                        [[4, 5, 0, 0],
                                         [4, 106, 5, 0],
                                         [4, 106, 5, 0]]]),
        'answer_utterance_cnt': tf.Variable([3, 3], dtype=tf.int32)
    }
    args = load_hyperparams()
    mask_id = 6
    boa_id = 4
    eoa_id = 5
    eos_id = 2
    pad_id = 0
    template_pack, answer_packs = prepare_template(data_batch, args, mask_id, pad_id)

    update_rst = []
    cur_template_pack = template_pack
    for hole in answer_packs:
        cur_template_pack = update_template_pack(cur_template_pack, hole['text_ids'][:, 1:], mask_id, eoa_id, pad_id)
        update_rst.append(cur_template_pack)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fetches = {
            'ori': data_batch,
            'template': template_pack,
            'fills': answer_packs,
            'updated': update_rst
        }
        rtns = sess.run(fetches)
        print(rtns['ori'])
        # print(rtns['updated'][-1])
        print(rtns['template'])
        print(rtns['fills'])
        for i in rtns['updated']:
            print('\n')
            print(i)
test_update_template_pack()
