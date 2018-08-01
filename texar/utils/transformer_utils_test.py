import numpy as np
import tensorflow as tf
from texar.utils.transformer_utils import generate_random_mask, generate_equal_length_mask, prepare_template, \
    _split_template, _merge_segments, fill_template


def test_generate_equal_length_mask():
    mask_length = 2
    mask_num = 2
    mask_id = 7
    eos_id = 8
    inputs = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int64)

    masks, answers, templates, _ = generate_equal_length_mask(inputs, lengths, mask_num, mask_length, mask_id, eos_id)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs_, masks_, answers_, templates_ = sess.run([inputs, masks, answers, templates])
        print(inputs_)
        print(masks_)
        print(answers_)
        print(templates_)
# test_generate_equal_length_mask()


def test_generate_random_mask():
    mask_id = 7
    eos_id = 8
    pad_id = 9
    present_rate = 0.4
    inputs = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                          [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int64)

    masks, answers, templates, _ = generate_random_mask(inputs, lengths, present_rate,
                                                        mask_id, eos_id, pad_id, batch_size=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs_, masks_, answers_, templates_ = sess.run([inputs, masks, answers, templates])
        print(inputs_)
        print(masks_)
        print(answers_)
        print(templates_)
# test_generate_random_mask()


def test_prepare_template():
    text_ids = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1],
                            [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    length = tf.Variable([11, 11], dtype=tf.int32)
    data_batch = {
        'text_ids': text_ids,
        'length': length
    }
    mask_num = 2
    mask_length = 3
    mask_id = 7
    eos_id = 8
    template_pack, answer_packs = \
        prepare_template(data_batch, mask_num, mask_length, mask_id, eos_id)

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
# test_fill_template_with_tensor()


def test_partition():
    num_partitions = 5
    data = tf.Variable([1, 2, 3, 4, 5, 6, 7])
    partitions = ([0, 0, 0, 1, 1, 2, 2])
    rst = tf.dynamic_partition(data=data,
                               partitions=partitions,
                               num_partitions=num_partitions)

    final_rst = []
    for r in rst:
        x = tf.cond(tf.equal(tf.size(r), 0), lambda: 1, lambda: 0)
        final_rst.append(x)

    rst1 = tf.dynamic_partition(data=tf.Variable(rst),
                                   partitions=tf.Variable(final_rst),
                                   num_partitions=2)
    print(rst)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rst_, rst1_, final_ = sess.run([rst, rst1, final_rst])
        print(rst_)
        print(rst1_)
        print(final_)

# test_partition()

def test_anther_partition():
    def dynamic_partition_png(vals, idx, max_partitions):
        """Encodes output of dynamic partition as a Tensor of png-encoded strings."""
        max_idx = tf.reduce_max(idx)
        max_vals = tf.reduce_max(idx)
        with tf.control_dependencies([tf.Assert(max_vals<256, ["vals must be <256"])]):
            outputs = tf.dynamic_partition(vals, idx, num_partitions=max_partitions)
        png_outputs = []
        dummy_png = tf.image.encode_png(([[[2]]]))
        not_empty_ops = [] # ops that detect empty lists that aren't at the end
        for i, o in enumerate(outputs):
            reshaped_o = tf.reshape(tf.cast(o, tf.uint8), [-1, 1, 1])
            png_output = tf.cond(tf.size(reshaped_o)>0, lambda: tf.image.encode_png(reshaped_o), lambda: dummy_png)
            png_outputs.append(png_output)
            not_empty_ops.append(tf.logical_or(i>max_idx, tf.size(reshaped_o)>0))
        packed_tensor = tf.stack(png_outputs)
        no_illegal_empty_lists = tf.reduce_all(tf.stack(not_empty_ops))
        with tf.control_dependencies([tf.Assert(no_illegal_empty_lists, ["empty lists must be last"])]):
            result = packed_tensor[:max_idx+1]
        return result

    def decode(p):
        return tf.image.decode_png(p)[:, 0, 0]

    vals = tf.constant([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    idx = [0, 1, 1, 1, 1]
    tf_vals = dynamic_partition_png(vals, idx, 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(decode(tf_vals[0])))  # => [1 2]
        print(sess.run(decode(tf_vals[1])))  # => [3 4 5]
        print(sess.run(decode(tf_vals[2])))  # => slice index 2 of dimension 0 out of bounds

# test_anther_partition()


def test_pad_answer():
    def _pad_answers(answers, lengths):
        """
        Pad the list of answers to the same size.
        :param answers: a list of tensors, each of shape [unfixed_len, batch_size].
                        Return value of dynamic_partition
        :return: a list of tensors of the same size
        """
        def _pad(ans, cur_len, max_len, dummy_ans):
            def _pad_py_func(ans, cur_len, max_len, dummy_ans):
                if np.size(ans) == 0:
                    return dummy_ans
                ans = np.pad(np.transpose(ans),
                             pad_width=((0, 0), (0, max_len-cur_len)),
                             mode='constant',
                             constant_values=((0, 0), (0, -1)))
                return ans.astype(np.int64)
            return tf.py_func(_pad_py_func,
                              [ans, cur_len, max_len, dummy_ans], tf.int64)

        def _get_dummy_ans(ans, max_len):
            def _get_dummy_ans_py_func(ans, max_len):
                batch_size = ans.shape[1]
                return np.zeros(shape=(batch_size, max_len), dtype=np.int64)
            return tf.py_func(_get_dummy_ans_py_func, [ans, max_len], tf.int64)

        padded_answers = []
        max_len = tf.reduce_max(lengths)
        dummy_ans = _get_dummy_ans(answers[0], max_len)
        for idx, ans in enumerate(answers):
            padded_ans = _pad(ans, lengths[idx], max_len, dummy_ans)
            padded_answers.append(padded_ans)
        return padded_answers

    answers = tf.Variable([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    partitions = tf.Variable([0, 0, 1, 1, 1])
    max_partition_num = 4
    answer_partitions = tf.dynamic_partition(data=tf.transpose(answers, perm=[1, 0]),  # [sum(lens), batch_size]
                                             partitions=partitions,
                                             num_partitions=max_partition_num)
    lengths = tf.Variable([2, 3, 0, 0])
    padded_redundant_answers = _pad_answers(answer_partitions, lengths)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r, p = sess.run([answer_partitions, padded_redundant_answers])
        print(r)
        print(p)

# test_pad_answer()


def test_random_fill_mask():
    def _fill_mask(inputs, lengths, present_rate, eoa_id, max_partition_num):
        """
        The input batch has the same mask pattern, randoms through max_seq_length in lengths.
        :param inputs:
        :param lengths:
        :param present_rate:
        :return: answers: a tensor of shape [batch_size, sum(unfixed_answer_len for each ans)]
        start_pos and end_pos marks out ranges for answers
        """

        def _fill_mask_py_func(inputs, lengths, present_rate, eoa_id, max_partition_num):
            seq_length = lengths.max()
            present_num = int(seq_length * present_rate)
            present_idx = np.sort(np.append(  # <BOS> has to be present
                np.random.choice(range(1, seq_length), present_num - 1, replace=False), 0))

            batch_size = inputs.shape[0]
            zeros = np.zeros(batch_size)
            eoa = np.full_like(zeros, eoa_id)[:, np.newaxis]
            masks = np.full_like(inputs, 1)
            for idx in present_idx:
                masks[:, idx] = zeros  # set present column to 0

            start_positions, end_positions = [], []
            lens = np.zeros(shape=(max_partition_num), dtype=np.int64)
            answers = np.array([[], []])
            partitions = np.array([])
            present_idx = np.append(present_idx, seq_length)  # add the last ending pos
            idx = 0
            for prev, cur in zip(present_idx[:-1], present_idx[1:]):
                if prev + 1 != cur:
                    start_positions.append(prev + 1)
                    end_positions.append(cur)
                    lens[idx] = cur - prev
                    cur_ans = np.concatenate((inputs[:, prev + 1:cur], eoa), axis=1)  # add eoa
                    answers = np.concatenate((answers, cur_ans), axis=1)
                    cur_idx = np.full_like(cur_ans[0], idx)
                    partitions = np.concatenate((partitions, cur_idx), axis=0)
                    idx += 1

            print("present_id:", present_idx)
            print("start_pos: ", start_positions)
            print("end_pos:   ", end_positions)
            print("lens:      ", lens)
            print("answers:   ", answers)
            start_positions = np.array(start_positions, dtype=np.int64)
            end_positions = np.array(end_positions, dtype=np.int64)
            return masks, start_positions, end_positions, answers.astype(np.int64), lens, partitions.astype(np.int64)

        eoa_id = tf.Variable(eoa_id, dtype=tf.int64)
        present_rate = tf.Variable(present_rate, dtype=tf.float32)
        return tf.py_func(_fill_mask_py_func,
                          [inputs, lengths, present_rate, eoa_id, max_partition_num],
                          [tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64])

    text_ids = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int32)
    present_rate = 0.8
    eoa_id = 99
    max_partition_num = 6
    masks, start_positions, end_positions, answers, ans_lens, partitions = \
        _fill_mask(text_ids, lengths, present_rate, eoa_id, max_partition_num)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        masks, start_positions, end_positions, answers, ans_lens, partitions = \
            sess.run([masks, start_positions, end_positions, answers, ans_lens, partitions])
        print("masks:\n", masks)
        print("answers:\n", answers)
        print("ans_lens:\n", ans_lens)
        print("partitions:\n", partitions)

# test_random_fill_mask()


def test_remove_pad():
    def _remove_pad(padded_answers, lengths):
        """
        Remove padding for each answer tensor in list.
        """
        answer = []
        for idx, padded_ans in enumerate(padded_answers):
            answer.append(padded_ans[:, :lengths[idx]])
        return answer

    padded_answers = [tf.Variable([[1, 2, 3, -1], [1, 2, 3, -1]]),
                      tf.Variable([[1, 2, -1, -1], [1, 2, -1, -1]])]
    lengths = tf.Variable([3, 2, 0, 0])
    rst = _remove_pad(padded_answers, lengths)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        r = sess.run(rst)
        print(r)

# test_remove_pad()


def test_generate_random_mask():
    inputs = tf.Variable([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int32)
    present_rate = 0.5
    mask_id = 99
    eoa_id = 22
    max_partition_num = 6
    masks, answers, templates, template_masks = \
        generate_random_mask(inputs, lengths, present_rate, mask_id, eoa_id, max_partition_num)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        masks, answers, templates, template_masks = sess.run([masks, answers, templates, template_masks])
        print("masks:\n", masks)
        print("answers:\n", answers)
        print("templates:\n", templates)
        print("template_masks:\n", template_masks)

test_generate_random_mask()