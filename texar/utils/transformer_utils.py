"""
This script is adapted from the tensor2tensor repositor.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from texar.utils.shapes import shape_list

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
        segment_ids: [[0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4],
                      [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4]] <- start from 0
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
            segment_ids[i][0] = 0
            for j in range(1, lengths[i]):
                if mask[j] == mask[j-1]:
                    segment_ids[i][j] = segment_ids[i][j-1]
                    offsets[i][j] = offsets[i][j-1] + 1
                else:
                    segment_ids[i][j] = segment_ids[i][j-1] + 1
                    offsets[i][j] = 0
        return segment_ids, offsets

    return tf.py_func(_parse_segment, [lengths, masks], [masks.dtype, masks.dtype])


def _pad_array_list(arrays, lens, pad_id):
    """
    :param ar: a list of 1-D array of different lengths, [batch_size, unfixed length]
    :return: a 2-D array, [batch_size, max_seq_len_in_original_list]
    """
    rst = []
    max_len = np.amax(lens)
    for idx, ar in enumerate(arrays):
        rst.append(np.pad(ar, (0, max_len - lens[idx]),
                          'constant', constant_values=pad_id))
    return np.array(rst), max_len


def _parse_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    """
    :param inputs:
    :param masks:
    :param start_positions: [batch_size, mask_num]
    :param end_positions:
    :param mask_id:
    :return:
    """
    inputs = inputs.tolist()
    masks = masks.tolist()
    l = len(inputs[0])
    rst, mask_rst, template_len = [], [], []
    for input, mask, start_pos_, end_pos_ in zip(inputs, masks, start_positions, end_positions):
        start_pos = [0]
        start_pos.extend(end_pos_.tolist())
        end_pos = start_pos_.tolist()
        end_pos.append(l)
        tmp_rst, tmp_mask = [], []
        for s, e in zip(start_pos, end_pos):
            tmp_rst.extend(input[s:e])
            tmp_rst.append(mask_id)
            tmp_mask.extend(mask[s:e])
            tmp_mask.append(1)
        tmp_rst.pop()  # delete the last mask_id
        tmp_mask.pop()
        rst.append(tmp_rst)
        mask_rst.append(tmp_mask)
        template_len.append(len(tmp_rst))
    rst, _ = _pad_array_list(rst, template_len, pad_id)
    mask_rst, _ = _pad_array_list(mask_rst, template_len, pad_id)
    return rst, mask_rst


def _prepare_squeezed_template(inputs, masks, start_positions, end_positions, mask_id, pad_id):
    templates, template_masks = \
        tf.py_func(_parse_template,
                   [inputs, masks, start_positions, end_positions, mask_id, pad_id],
                   [tf.int64, tf.int64])
    batch_size = tf.shape(inputs)[0]
    templates = tf.reshape(templates, shape=tf.stack([batch_size, -1]))
    template_masks = tf.reshape(template_masks, shape=tf.stack([batch_size, -1]))
    return templates, template_masks


def generate_prediction_offsets(inputs, max_length):
    batch_size = tf.shape(inputs)[0]
    max_length = tf.cast(max_length, dtype=tf.int32)
    _, offsets = parse_segment(tf.fill([batch_size], max_length),
                               tf.fill([batch_size, max_length], 0))
    return tf.cast(offsets, dtype=tf.int64)


def generate_prediction_segment_ids(inputs, segment_id, max_length):
    batch_size = tf.shape(inputs)[0]
    return tf.cast(tf.fill([batch_size, tf.cast(max_length, dtype=tf.int32)], segment_id), dtype=tf.int64)


def prepare_template(data_batch, args, mask_id, pad_id):
    """
    data_batch:
    {'source_text': array([[b'<BOS>', b'and', b'she', b'sprang', b'off', b'his', b'shoulder', b'and', b'up', b'the', b'steps', b'before', b'him', b'<EOS>'],
                           [b'<BOS>', b'and', b'they', b'gave', b'hans', b'gifts', b'of',b'gold', b'and', b'of', b'silver', b'<EOS>', b'', b'']], dtype=object),
    'source_length': array([14, 12], dtype=int32),
    'source_text_ids': array([[   1,   10,   47, 1068,   44,  166, 1990,   10,  287,   49, 1401, 143,  115,    2],
                              [   1,   10,   19,   48, 1913,  775,  106,  778,   10,  106,  477, 2,    0,    0]]),
    'templatebyword_text': array([[b'<BOS>', b'and', b'she', b'sprang', b'<m>', b'his', b'shoulder', b'and', b'<m>', b'<m>', b'steps', b'<m>', b'him', b'<EOS>'],
                                  [b'<BOS>', b'and', b'they', b'gave', b'<m>', b'hans', b'gifts', b'<m>', b'gold', b'and', b'<m>', b'silver', b'<EOS>', b'']], dtype=object),
    'templatebyword_length': array([14, 13], dtype=int32),
    'templatebyword_text_ids': array([[   1,   10,   47, 1068,    6,  166, 1990,   10,    6,    6, 1401, 6,  115,    2],
                                      [   1,   10,   19,   48,    6, 1913,  775,    6,  778,   10,    6, 477,    2,    0]]),
    'answer_text': array([[[b'<BOA>', b'off', b'<EOA>', b'<PAD>'],
                           [b'<BOA>', b'up', b'the', b'<EOA>'],
                           [b'<BOA>', b'before', b'<EOA>', b'<PAD>']],
                           [[b'<BOA>', b'<EOA>', b'<PAD>', b''],
                            [b'<BOA>', b'of', b'<EOA>', b''],
                            [b'<BOA>', b'of', b'<EOA>', b'']]], dtype=object),
    'answer_length': array([[3, 4, 3], [2, 3, 3]], dtype=int32),
    'answer_text_ids': array([[[  4,  44,   5,   0],
                                [  4, 287,  49,   5],
                                [  4, 143,   5,   0]],
                               [[  4,   5,   0,   0],
                                [  4, 106,   5,   0],
                                [  4, 106,   5,   0]]]),
    'answer_utterance_cnt': array([3, 3], dtype=int32)}
    """
    def _get_start_end_pos(mask_by_word, mask_id):
        def _get_start_end_pos_py_func(mask_by_word, mask_id):
            start_pos, end_pos = [[-2] for i in range(len(mask_by_word))], [[-2] for i in range(len(mask_by_word))]
            for idx, template in enumerate(mask_by_word):
                for i, word in enumerate(template):
                    if word == mask_id:
                        if start_pos[idx][-1] == i - 1:
                            end_pos[idx].pop()
                        else:
                            start_pos[idx].append(i)
                        end_pos[idx].append(i+1)
            return np.array(start_pos)[:, 1:].astype(np.int64), np.array(end_pos)[:, 1:].astype(np.int64)

        mask_id = tf.Variable(mask_id, dtype=tf.int64)
        return tf.py_func(_get_start_end_pos_py_func,
                          [mask_by_word, mask_id],
                          [tf.int64, tf.int64])

    masked_inputs = data_batch['templatebyword_text_ids']
    masks = tf.where(tf.equal(masked_inputs, mask_id * tf.ones_like(masked_inputs)),
                     tf.ones_like(masked_inputs), tf.zeros_like(masked_inputs))
    start_positions, end_positions = _get_start_end_pos(masked_inputs, mask_id)
    templates, template_masks = \
        _prepare_squeezed_template(masked_inputs, masks, start_positions, end_positions, mask_id, pad_id)
    template_lengths = tf.fill(tf.shape(data_batch['source_length']), tf.shape(templates)[1])
    template_segment_ids, template_offsets = \
        parse_segment(template_lengths, template_masks)
    template_pack = {
        'text_ids': masked_inputs,
        'segment_ids': template_segment_ids,
        'offsets': template_offsets,
        'templates': templates,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'masks': masks,
        'template_lengths': template_lengths
    }

    answers = tf.dynamic_partition(
        data=tf.transpose(data_batch['answer_text_ids'], perm=[1, 0, 2]),
        partitions=list(range(args.partition_num)),
        num_partitions=args.partition_num
    )
    answer_lengths = tf.dynamic_partition(
        tf.transpose(data_batch['answer_length'], perm=[1, 0]),
        partitions=list(range(args.partition_num)),
        num_partitions=args.partition_num
    )
    answer_packs = []
    for idx, (answer, answer_length)in enumerate(zip(answers, answer_lengths)):
        answer = answer[0]
        answer_length = answer_length[0]
        mask_len = shape_list(answer)[1]
        answer_segment_ids = generate_prediction_segment_ids(answer, idx * 2 + 1, mask_len)
        answer_offsets = generate_prediction_offsets(answer, mask_len)
        answer_packs.append({
            'text_ids': answer,
            'segment_ids': answer_segment_ids,
            'offsets': answer_offsets,
            'lengths': answer_length
        })
    return template_pack, answer_packs


def _split_template(template, mask_start_positions, mask_end_positions):
    """
    template: [3, 5, 4, 7, 7, 1, 3, 3, 7, 7, 1]
    start_positions: [3, 8], starting positions of the masks
    end_positions: [5, 10], ending positions of the masks
    will be split into: [[3, 5, 4], [1, 3, 3], [1]]
    :param template: a list of numbers
    :return:
    """
    rst = []
    start_positions = [0] + mask_end_positions.tolist()
    end_positions = mask_start_positions.tolist() + [len(template)]
    for s, e in zip(start_positions, end_positions):
        rst.append(template[s: e])
    return rst


def _merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id):
    """
    template_segments: [[3, 5, 4], [1, 3, 3], [1]]
    fillings: [[4, 2], [2, 5]]
    rst: [3, 5, 4, 4, 2, 1, 3, 3, 2, 5, 1]
    :param template_segments:
    :param fillings:
    :return:
    """
    def _parse(id_list, eoa_id, pad_id, eos_id):
        rst = []
        for id in id_list:
            if id in [eoa_id, eos_id]:
                break
            elif id is not pad_id:
                rst.append(id)
        return rst

    template_segment_num = len(template_segments)
    filling_segment_num = len(fillings)
    assert template_segment_num == filling_segment_num or \
           template_segment_num == filling_segment_num + 1

    rst = []
    for i in range(filling_segment_num):
        rst.extend(template_segments[i])
        rst.extend(_parse(fillings[i], eoa_id, pad_id, eos_id))
    if template_segment_num > filling_segment_num:
        rst.extend(template_segments[-1])
    return rst


def fill_template(template_pack, predictions, eoa_id, pad_id, eos_id):
    """
    :param template: [batch_size, max_seq_len]
    :param mask: [batch_size, max_seq_len]
    :param predictions: a list of tensors
    :return:
    """
    def _transpose(a):
        """
        :param a: mask_num * batch_size * undefined_len
        :return: batch_size * mask_num * undefined_len
        """
        rst = [[] for _ in a[0]]
        for ar in a:
            for idx, sent in enumerate(ar):
                rst[idx].append(sent)
        return rst

    start_positions = template_pack['start_positions']
    end_positions = template_pack['end_positions']
    templates = template_pack['text_ids']
    templates = templates.tolist()
    predictions = [prediction.tolist() for prediction in predictions]  # mask_num * batch_size * undefined_len
    predictions = _transpose(predictions)
    rst = []
    for template, start_pos, end_pos, fillings in zip(templates, start_positions, end_positions, predictions):
        template_segments = _split_template(template, start_pos, end_pos)
        rst.append(_merge_segments(template_segments, fillings, eoa_id, pad_id, eos_id))
    return rst
