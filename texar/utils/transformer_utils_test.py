import numpy as np
import tensorflow as tf
from texar.utils.transformer_utils import generate_mask, prepare_template, \
    _split_template, _merge_segments, fill_template


def test_generate_mask():
    mask_length = 2
    mask_num = 2
    mask_id = 7
    eos_id = 8
    inputs = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int64)

    masks, answers, templates, _ = generate_mask(inputs, lengths, mask_num, mask_length, mask_id, eos_id)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        inputs_, masks_, answers_, templates_ = sess.run([inputs, masks, answers, templates])
        print(inputs_)
        print(masks_)
        print(answers_)
        print(templates_)


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
test_fill_template_with_tensor()
