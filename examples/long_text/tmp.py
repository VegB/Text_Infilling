import tensorflow as tf
import texar as tx
import numpy as np
from texar.data import SpecialTokens

def test_mono():
    hyp = {
            "num_epochs": 50,
            "batch_size": 3,
            "dataset": {
                "files": ['./grimm_data/grimm.answer.test.txt'],
                "vocab_file": "./grimm_data/vocab.txt",
                "delimiter": " ",
                "max_seq_length": None,
                "length_filter_mode": "truncate",
                "pad_to_max_seq_length": False,
                "bos_token": SpecialTokens.BOA,
                "eos_token": SpecialTokens.EOA,
                "other_transformations": [],
                "variable_utterance": True,
                "utterance_delimiter": "|||"
            }
    }

    answer_data = tx.data.MonoTextData(hyp)
    iterator = tx.data.DataIterator(answer_data)
    data_batch = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        iterator.switch_to_dataset(sess)
        print(sess.run(data_batch))


def test_get_start_end_pos():
    def _get_start_end_pos_py_func(mask_by_word, mask_id):
        start_pos, end_pos = [[-2] for i in range(len(mask_by_word))], [[-2] for i in range(len(mask_by_word))]
        for idx, template in enumerate(mask_by_word):
            for i, word in enumerate(template):
                if word == mask_id:
                    if start_pos[idx][-1] == i - 1:
                        end_pos[idx].pop()
                    else:
                        start_pos[idx].append(i)
                    end_pos[idx].append(i + 1)
        return np.array(start_pos)[:, 1:], np.array(end_pos)[:, 1:]

    mask_by_word = [[1,   10,   47, 1068,    6,  166, 1990,   10,    6,    6, 1401, 6,  115,    2],
                    [1,   10,   19,   48,    6, 1913,  775,    6,  778,   10,    6, 477,    2,    0]]
    print(_get_start_end_pos_py_func(mask_by_word, mask_id=6))


if __name__ == '__main__':
    #test_get_start_end_pos()
    test_mono()
