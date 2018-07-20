"""
This script is adapted from the tensor2tensor repositor.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

__all__ = [
    "PadRemover",
    "embedding_to_padding",
    "_bucket_boundaries",
    "_batching_scheme",
    "smoothing_cross_entropy",
    "prepare_template",
    "fill_template",
    "generate_prediction_offsets",
    "generate_prediction_segment_ids"
]


class PadRemover(object):
    """Helper to remove padding from a tensor before sending to the experts.
    The padding is computed for one reference tensor containing the padding mask
    and then can be applied to any other tensor of shape [dim_origin,...].
    Ex:
            input = [
                [tok1, tok2],
                [tok3, tok4],
                [0, 0],
                [0, 0],
                [tok5, tok6],
                [0, 0],
            ]
            output = [
                [tok1, tok2],
                [tok3, tok4],
                [tok5, tok6],
            ]
    """

    def __init__(self, pad_mask):
        """Compute and store the location of the padding.
        Args:
            pad_mask (tf.Tensor): Reference padding tensor of shape
                [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
                containing non-zeros positive values to indicate padding location.
        """
        self.nonpad_ids = None
        self.dim_origin = None

        with tf.name_scope("pad_reduce/get_ids"):
            pad_mask = tf.reshape(pad_mask, [-1])    # Flatten the batch
            # nonpad_ids contains coordinates of zeros rows (as pad_mask is
            # float32, checking zero equality is done with |x| < epsilon, with
            # epsilon=1e-9 as standard, here pad_mask only contains positive values
            # so tf.abs would be redundant)
            self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
            self.dim_origin = tf.shape(pad_mask)[:1]

    def remove(self, x):
        """Remove padding from the given tensor.
        Args:
            x (tf.Tensor): of shape [dim_origin,...]
        Returns:
            a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
        """
        with tf.name_scope("pad_reduce/remove"):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(
                    x,
                    indices=self.nonpad_ids,
            )
            #if not context.in_eager_mode():
            # This is a hack but for some reason, gather_nd return a tensor of
            # undefined shape, so the shape is set up manually
            x.set_shape([None] + x_shape[1:])
        return x

    def restore(self, x):
        """Add padding back to the given tensor.
        Args:
            x (tf.Tensor): of shape [dim_compressed,...]
        Returns:
            a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
            dim is restored from the original reference tensor
        """
        with tf.name_scope("pad_reduce/restore"):
            x = tf.scatter_nd(
                    indices=self.nonpad_ids,
                    updates=x,
                    shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
            )
        return x

def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.
    We have hacked symbol_modality to return all-zero embeddings for padding.
    Args:
        emb: a Tensor with shape [..., depth].
    Returns:
        a float Tensor with shape [...].
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))
def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def _batching_scheme(batch_size,
                     max_length,
                     min_length_bucket,
                     length_bucket_step,
                     drop_long_sequences=False,
                     shard_multiplier=1,
                     length_multiplier=1,
                     min_length=0,
                     batch_relax=False):
  """A batching scheme based on model hyperparameters.
  Every batch containins a number of sequences divisible by `shard_multiplier`.
  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.
  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example
  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-compoisite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3

  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret

def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False,
                            zero_pad=True):
    """Cross entropy with label smoothing to limit over-confidence.
    Args:
        logits: Tensor of size [batch_size, ?, vocab_size]
        labels: Tensor of size [batch_size, ?]
        vocab_size: Tensor representing the size of the vocabulary.
        confidence: Used to determine on and off values for label smoothing.
            If `gaussian` is true, `confidence` is the variance to the gaussian
            distribution.
        gaussian: Uses a gaussian distribution for label smoothing
        zero_pad: use 0 as the probabitlity of the padding
            in the smoothed labels. By setting this, we replicate the
            numeric calculation of tensor2tensor, which doesn't set the
            <BOS> token in the vocabulary.
    Returns:
        the cross entropy loss.
    """
    with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
        # Low confidence is given to all non-true labels, uniformly.
        if zero_pad:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 2)
        else:
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)

        if gaussian and confidence > 0.0:
            labels = tf.cast(labels, tf.float32)
            normal_dist = tf.distributions.Normal(loc=labels, scale=confidence)
            soft_targets = normal_dist.prob(
                tf.cast(tf.range(vocab_size), tf.float32)\
                    [:, None, None])
            # Reordering soft_targets from [vocab_size, batch_size, ?]
            # to match logits: [batch_size, ?, vocab_size]
            soft_targets = tf.transpose(soft_targets, perm=[1, 2, 0])
        else:
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence,
                dtype=logits.dtype)
        if zero_pad:
            soft_targets = tf.concat([tf.expand_dims(\
                tf.zeros_like(labels, dtype=tf.float32), 2),\
                soft_targets[:, :, 1:]], -1)

        if hasattr(tf.nn, 'softmax_cross_entropy_with_logits_v2'):
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
        else:
            cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
        logits=logits, labels=soft_targets)


def parse_segment(lengths, masks):
    def _parse_segment(lengths, masks):
        """
        mask:        [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
        segment_ids: [[1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                      [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5]]
        offsets:     [[0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0],
                      [0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0]]
        :param masks:
        :return: segment_ids, offsets
        """
        segment_ids = np.full_like(masks, 0)
        offsets = np.full_like(masks, 0)
        batch_size = masks.shape[0]
        for i in range(batch_size):
            mask = masks[i]
            segment_ids[i][0] = 1
            for j in range(1, lengths[i]):
                if mask[j] == mask[j-1]:
                    segment_ids[i][j] = segment_ids[i][j-1]
                    offsets[i][j] = offsets[i][j-1] + 1
                else:
                    segment_ids[i][j] = segment_ids[i][j-1] + 1
                    offsets[i][j] = 0
        return segment_ids, offsets

    return tf.py_func(_parse_segment, [lengths, masks], [masks.dtype, masks.dtype])


def generate_mask(inputs, lengths, mask_num, mask_length):
    """
    inputs and lengths are numpy arrays!
    mask_num = 2, having two masked out segment
    mask_length = 2, min length of each masked out segment
    inputs:[[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]]
    mask:  [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
    :param inputs: 
    :param mask_num: 
    :param mask_length: 
    :return: 
    """
    def _parse_answer(inputs, start_pos, end_pos):
        rst = None
        batch_size = inputs.shape[0]
        for i in range(batch_size):
            tmp_answer = inputs[i, start_pos[i]:end_pos[i]][np.newaxis, :]
            rst = tmp_answer if rst is None else np.concatenate((rst, tmp_answer), axis=0)
        return rst

    # TODO(wanrong): OUT-OF-RANGE bound check, tf.argmin(length) >= masknum * (1 + mask_length)
    def _fill_mask(mask_not_generated, mask_length, prev_end_pos, lengths, masks):
        """
        :param mask_not_generated: number of mask not generated(excluding this one)
        :param prev_end_pos: open range
        :param masks:
        :return: cur_end_pos, updated masks
        """
        cur_start_pos = np.full_like(prev_end_pos, 1)
        batch_size = cur_start_pos.shape[0]
        for i in range(batch_size):
            cur_start_pos[i] = \
                np.random.randint(prev_end_pos[i] + 1,
                                  lengths[i] - mask_not_generated * (1 + mask_length) + 1,
                                  size=1)
        cur_end_pos = cur_start_pos + mask_length
        for i in range(batch_size):
            masks[i, cur_start_pos[i]: cur_end_pos[i]] = 1
        return mask_not_generated - 1, mask_length, cur_end_pos, lengths, masks
    mask_not_generated = tf.Variable(mask_num, dtype=tf.int64)
    mask_length = tf.Variable(mask_length, dtype=tf.int64)
    prev_end_pos = tf.ones_like(inputs)[:, 0]
    masks = tf.zeros_like(inputs)
    answers = []  # None
    for i in range(mask_num):
        mask_not_generated, _, prev_end_pos, _, masks = \
            tf.py_func(_fill_mask,
                       [mask_not_generated, mask_length, prev_end_pos, lengths, masks],
                       [tf.int64, tf.int64, prev_end_pos.dtype, lengths.dtype, mask_length.dtype])
        cur_answer = tf.py_func(_parse_answer,
                                [inputs, prev_end_pos - mask_length, prev_end_pos],
                                inputs.dtype)
        answers.append(cur_answer)
    return masks, answers


def test_generate_mask():
    mask_length = 2
    mask_num = 2
    inputs = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    lengths = tf.Variable([11, 11], dtype=tf.int64)

    masks, answers = generate_mask(inputs, lengths, mask_num, mask_length)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        masks_, answers_ = sess.run([masks, answers])
        print(masks_)
        print(answers_)
        print(answers_[:, 0])


def prepare_template(data_batch, mask_num, mask_length, mask_id):
    """
    mask_id = 7
    pad_id = 6
    inputs:        [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]] <- a tensor
    mask:          [[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]] <- 1 is masked out
    masked_inputs: [[3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1], [2, 1, 7, 7, 7, 7, 5, 4, 7, 7, 5]] <- template
    segment_ids:   [[1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5], [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5]]
    answers:       [[[4, 2, 6, 6], [2, 5, 6, 6]], 
                    [[4, 3, 5, 1], [3, 1, 6, 6]]] <- used as decode outputs(targets) in training
    :param masked_inputs:
    :param mask_id:
    :return: masked_inputs, segment_ids, answers
    """
    inputs = data_batch['text_ids']
    lengths = data_batch['length']
    masks, answers = generate_mask(inputs, lengths, mask_num, mask_length)
    template_segment_ids, template_offsets = parse_segment(lengths, masks)
    all_masked_out = tf.cast(tf.fill(tf.shape(inputs), mask_id), dtype=tf.int64)
    masked_inputs = tf.where(tf.equal(masks, tf.ones_like(inputs)),
                             all_masked_out, inputs)
    template_pack = {
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'masks': masks
    }

    answer_packs = []
    for answer in answers:
        answer_segment_ids, answer_offsets =\
            parse_segment(tf.fill(tf.shape(lengths), mask_length),
                          tf.zeros_like(answer))
        answer = tf.reshape(answer, shape=tf.stack([-1, mask_length]))
        answer_packs.append({
            'text_ids': answer,
            'segment_ids': answer_segment_ids,
            'offsets': answer_offsets
        })

    return template_pack, answer_packs


def test_prepare_template():
    text_ids = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    length = tf.Variable([11, 11], dtype=tf.int32)
    data_batch = {
        'text_ids': text_ids,
        'length': length
    }
    mask_num = 2
    mask_length = 3
    mask_id = 7
    template_pack, answer_packs = prepare_template(data_batch, mask_num, mask_length, mask_id)

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


def _split_template(template, mask_id):
    """
    template: [3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1]
    will be split into: [[3, 5, 4], [1, 3, 3], [1]]
    :param template: a list of numbers
    :return:
    """
    def _find(eles, tgt):
        for idx, ele in enumerate(eles):
            if ele == tgt:
                for i, e in enumerate(eles[idx:]):
                    if e != tgt:
                        return idx, eles[:idx], eles[idx+i:]
                return idx, eles[:idx], None
        return -1, None, eles

    rst = []
    pos, segment, template = _find(template, mask_id)
    while pos != -1 and template is not None:
        rst.append(segment)
        pos, segment, template = _find(template, mask_id)
    rst.append(template if template is not None else segment)
    return rst


def _merge_segments(template_segments, fillings):
    """
    template_segments: [[3, 5, 4], [1, 3, 3], [1]]
    fillings: [[4, 2], [2, 5]]
    rst: [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    :param template_segments:
    :param fillings:
    :return:
    """
    template_segment_num = len(template_segments)
    filling_segment_num = len(fillings)
    assert template_segment_num == filling_segment_num or \
           template_segment_num == filling_segment_num + 1

    rst = []
    for i in range(filling_segment_num):
        rst.extend(template_segments[i])
        rst.extend(fillings[i])
    if template_segment_num > filling_segment_num:
        rst.extend(template_segments[-1])
    return rst


def fill_template(templates, predictions, mask_id):
    """
    :param template: [batch_size, max_seq_len]
    :param mask: [batch_size, max_seq_len]
    :param predictions: a list of tensors
    :return:
    """
    templates = templates.tolist()
    predictions = np.array([prediction.tolist() for prediction in predictions])
    predictions = np.transpose(predictions, (1, 0, 2))
    rst = []
    for template, fillings in zip(templates, predictions):
        template_segments = _split_template(template, mask_id)
        rst.append(_merge_segments(template_segments, fillings))
    return rst


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
    assert rst == [[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 6, 2, 5, 3, 1, 4, 5]]


def test_fill_template_with_tensor():
    text_ids = tf.Variable([[3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1], [2, 1, 4, 3, 5, 1, 5, 4, 3, 1, 5]], dtype=tf.int64)
    length = tf.Variable([11, 11], dtype=tf.int32)
    data_batch = {
        'text_ids': text_ids,
        'length': length
    }
    mask_num = 3
    mask_length = 2
    mask_id = 7
    template_pack, answer_packs = prepare_template(data_batch, mask_num, mask_length, mask_id)

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

        filled = fill_template(rtns['template']['text_ids'], predictions, mask_id)
        print(filled)
        assert filled == rtns['ori']['text_ids'].tolist()


def generate_prediction_offsets(inputs, max_length):
    batch_size = tf.shape(inputs)[0]
    _, offsets = parse_segment(tf.fill([batch_size], max_length),
                               tf.fill([batch_size, max_length], 0))
    return tf.cast(offsets, dtype=tf.int64)


def generate_prediction_segment_ids(inputs, segment_id, max_length):
    batch_size = tf.shape(inputs)[0]
    return tf.cast(tf.fill([batch_size, max_length], segment_id), dtype=tf.int64)
