import argparse
import numpy as np
import tensorflow as tf
from texar.utils.transformer_utils import generate_random_mask, generate_equal_length_mask,\
    prepare_template, _split_template, _merge_segments, fill_template



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
    args.max_partition_num = int((args.max_seq_length + 1) / 2)
    return args


def test_generate_equal_length_mask():
    mask_length = 2
    mask_num = 2
    mask_id = 7
    eos_id = 8
    inputs = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                          [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int64)

    masks, answers, templates, _ = \
        generate_equal_length_mask(inputs, lengths, mask_num, mask_length, mask_id, eos_id)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs_, masks_, answers_, templates_ = sess.run([inputs, masks, answers, templates])
        print(inputs_)
        print(masks_)
        print(answers_)
        print(templates_)
# test_generate_equal_length_mask()


def test_prepare_template():
    inputs = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int64)
    length = tf.Variable([11, 11], dtype=tf.int32)
    data_batch = {
        'text_ids': inputs,
        'length': length
    }
    mask_id = 22
    eoa_id = 99
    args = load_hyperparams()
    template_pack, answer_packs = \
        prepare_template(data_batch, args, mask_id, eoa_id)

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
    b = [3, 2, 7, 7]

    assert _split_template(a, 7) == [[3, 5, 4], [1, 3, 3], [1]]
    assert _split_template(b, 7) == [[3, 2]]


def test_merge_segments():
    t_seg_1 = [[3, 5, 4], [1, 3, 3], [1]]
    fillings_1 = [[4, 2], [2, 5]]
    assert _merge_segments(t_seg_1, fillings_1) == \
           [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    assert _merge_segments(t_seg_1[:-1], fillings_1) == \
           [3, 5, 4, 4, 2, 1, 3, 3, 2, 5]


def test_fill_template():
    templates = np.array([[3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1],
                          [2, 1, 7, 7, 6, 2, 5, 7, 7, 4, 5]])
    predictions = np.array([[[4, 2], [2, 5]], [[4, 3], [3, 1]]])
    mask_id = 7
    rst = fill_template(templates, predictions, mask_id)
    assert rst == [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                   [2, 1, 4, 3, 6, 2, 5, 3, 1, 4, 5]]


def test_fill_template_with_tensor():
    text_ids = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                            [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    length = tf.Variable([11, 11], dtype=tf.int32)
    data_batch = {
        'text_ids': text_ids,
        'length': length
    }
    mask_num = 3
    mask_length = 2
    max_decode_len = 6
    mask_id = 7
    eos_id = 8
    template_pack, answer_packs = \
        prepare_template(data_batch, mask_num, mask_length,
                         max_decode_len, mask_id, eos_id)

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
            predictions.append(hole['text_ids'])

        filled = fill_template(rtns['template']['text_ids'], predictions, mask_id, eos_id)
        print(filled)
        assert filled == rtns['ori']['text_ids'].tolist()


def test_generate_random_mask():
    inputs = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int32)
    present_rate = 0.5
    mask_id = 99
    eoa_id = 22
    max_partition_num = 6
    masks, answers, ans_len, templates, template_masks = \
        generate_random_mask(inputs, lengths, present_rate,
                             mask_id, eoa_id, max_partition_num)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        masks, answers, templates, template_masks = \
            sess.run([masks, answers, templates, template_masks])
        print("masks:\n", masks)
        print("answers:\n", answers)
        print("templates:\n", templates)
        print("template_masks:\n", template_masks)


def test_random_fill_mask():
    def _fill_mask(inputs, lengths, present_rate, eoa_id, partition_num):
        """
        The input batch has the same mask pattern, randoms through max_seq_length in lengths.
        :param inputs:
        :param lengths:
        :param present_rate:
        :return: answers: a tensor of shape [batch_size, sum(unfixed_answer_len for each ans)]
        start_pos and end_pos marks out ranges for answers
        """

        def _fill_mask_py_func(inputs, lengths, present_rate, eoa_id, partition_num):
            # TODO(wanrong): bound check
            seq_length = lengths.max()
            masked_num = seq_length - int(seq_length * present_rate)

            # split masked_num into partition_num segments
            split_positions = \
                np.random.choice(range(1, masked_num - 1), partition_num - 1, replace=False)
            split_positions = np.sort(np.insert(np.insert(split_positions, 0, 0, axis=0),
                                                partition_num, masked_num, axis=0))

            # calculate the length of each mask segment
            mask_lengths = np.zeros(shape=partition_num, dtype=np.int64)  # add a 0 at the end
            for idx, (prev, cur) in enumerate(zip(split_positions[:-1], split_positions[1:])):
                mask_lengths[idx] = cur - prev
            print("mask len: ", mask_lengths)
            left_len = np.zeros(shape=partition_num + 1, dtype=np.int64)
            left_len[-1] = -1
            for idx, cur_len in reversed(list(enumerate(mask_lengths))):
                print("idx: %d, cur_len: %d" % (idx, cur_len))
                left_len[idx] = left_len[idx+1] + cur_len + 1
            left_len = left_len[:-1]

            print('left len: ', left_len)
            print('seq len:  ', seq_length)
            # splitting
            batch_size = inputs.shape[0]
            ones = np.ones(batch_size)
            eoa = np.full_like(ones, eoa_id)[:, np.newaxis]
            start_positions, end_positions = [0], [0]
            answers = np.array([[], []])
            partitions = np.array([])
            masks = np.full_like(inputs, 0)
            for i in range(1, partition_num + 1):
                idx = i - 1  # ignore padding 0 in start/end_positions
                # get start and end position for current mask
                print("low: %d, high: %d" % (end_positions[idx] + 1, seq_length - left_len[idx]))
                cur_start_pos = \
                    np.random.randint(end_positions[idx] + 1, seq_length - left_len[idx] + 1, size=1)[0]
                cur_end_pos = cur_start_pos + mask_lengths[idx]
                start_positions.append(cur_start_pos)
                end_positions.append(cur_end_pos)
                print("start pos: %d, end pos: %d" % (cur_start_pos, cur_end_pos))

                # get current answer
                cur_ans = np.concatenate(
                    (inputs[:, cur_start_pos: cur_end_pos], eoa), axis=1)  # add eoa
                answers = np.concatenate((answers, cur_ans), axis=1)

                # generate current partition index
                cur_idx = np.full_like(cur_ans[0], idx)
                partitions = np.concatenate((partitions, cur_idx), axis=0)

                # update mask
                for j in range(cur_start_pos, cur_end_pos):
                    masks[:, j] = ones  # set masked column to 1

            def _list_to_tiled_array(l):
                return np.tile(np.array(l, dtype=np.int64)[np.newaxis, :],
                               reps=(batch_size, 1))

            start_positions = _list_to_tiled_array(start_positions[1:])
            end_positions = _list_to_tiled_array(end_positions[1:])

            print("batch_size:\n", batch_size)
            print("start_pos:\n", start_positions)
            print("end_pos:\n", end_positions)
            print("mask_lens:\n", mask_lengths)
            print("masks:\n", masks)
            print("answers:\n", answers)
            print("parititions:\n", partitions)

            return masks, start_positions, end_positions, answers.astype(np.int64), \
                   mask_lengths, partitions.astype(np.int32)

        eoa_id = tf.Variable(eoa_id, dtype=tf.int64)
        present_rate = tf.Variable(present_rate, dtype=tf.float32)
        partition_num = tf.Variable(partition_num, dtype=tf.int64)
        return tf.py_func(_fill_mask_py_func,
                          [inputs, lengths, present_rate, eoa_id, partition_num],
                          [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int32])

    inputs = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int32)
    present_rate = 0.5
    mask_id = 99
    eoa_id = 22
    partition_num = 5

    masks, start_positions, end_positions, answers, ans_lens, partitions = \
        _fill_mask(inputs, lengths, present_rate, eoa_id, partition_num)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        masks, start_positions, end_positions, answers, ans_lens, partitions = \
            sess.run([masks, start_positions, end_positions, answers, ans_lens, partitions])

test_random_fill_mask()