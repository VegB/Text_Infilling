import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS


def prepare_data_batch(config, data_batch):
    """
    :param data_batch["text_ids"]: (batch_size, seq_len)
    :return: mask, encoder_input, decoder_input,
            decoder_output: (batch_size, seq_len-1)
    """
    real_ids = data_batch["text_ids"]

    mask = generate_mask(real_ids, config.is_present_rate)
    real_inputs = real_ids[:, :-1]
    masked_inputs = transform_input_with_is_missing_token(real_inputs, mask)

    return mask, masked_inputs, real_inputs, real_ids[:, 1:]


def generate_mask(real_ids, is_present_rate):
    """
    Generate the mask to be fed into the model.
    """
    batch_size = tf.shape(real_ids)[0]
    sequence_length = tf.shape(real_ids)[1] - 1
    
    if FLAGS.mask_strategy == 'random':
        p = np.random.choice([True, False], size=[batch_size, sequence_length],
            p=[is_present_rate, 1. - is_present_rate])

    elif FLAGS.mask_strategy == 'contiguous':
        masked_length = int((1 - is_present_rate) * sequence_length) - 1

        # Determine location to start masking.
        start_mask = np.random.randint(
            1, sequence_length - masked_length + 1, size=batch_size)
        p = np.full([batch_size, sequence_length], True, dtype=bool)

        # Create contiguous masked section to be False.
        for i, index in enumerate(start_mask):
          p[i, index:index + masked_length] = False

    else:
        raise NotImplementedError

    return p


def transform_input_with_is_missing_token(inputs, targets_present):
    """
    Transforms the inputs to have missing tokens when it's masked out.  The
    mask is for the targets, so therefore, to determine if an input at time t is
    masked, we have to check if the target at time t - 1 is masked out.
    e.g.
    inputs = [a, b, c, d]
    targets = [b, c, d, e]
    targets_present = [1, 0, 1, 0]
    then,
    transformed_input = [a, b, <missing>, d]
    Args:
    inputs:  tf.int32 Tensor of shape [batch_size, sequence_length] with tokens
      up to, but not including, vocab_size.
    targets_present:  tf.bool Tensor of shape [batch_size, sequence_length] with
      True representing the presence of the word.
    Returns:
    transformed_input:  tf.int32 Tensor of shape [batch_size, sequence_length]
      which takes on value of inputs when the input is present and takes on
      value=vocab_size to indicate a missing token.
    """
    batch_size = tf.shape(inputs)[0]
    sequence_length = tf.shape(inputs)[1]
    
    # To fill in if the input is missing.
    input_missing = tf.constant(0, dtype=tf.int32,
                                shape=[batch_size, sequence_length])

    # The 0th input will always be present.
    zeroth_input_present = tf.constant(True, tf.bool, shape=[batch_size, 1])

    # Input present mask.
    inputs_present = tf.concat(
        [zeroth_input_present, targets_present[:, :-1]], axis=1)

    transformed_input = tf.where(inputs_present, inputs, input_missing)
    return transformed_input
