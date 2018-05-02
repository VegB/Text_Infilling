import numpy as np
from gensim.models import KeyedVectors


def create_word_embedding(vocabs, emb_dim=50, vecfile=None):
    """
    :param vocabs: a dictionary. {key: word, val: idx}
    :param emb_dim:
    :param vecfile:
    :return:
    """
    emb = np.zeros(shape=(len(vocabs), emb_dim), dtype=np.float32)
    if vecfile is None:
        return emb

    pretrained = KeyedVectors.load_word2vec_format(vecfile, binary=True)

    def word_to_vec(word):
        if word in pretrained.vocab:
            return pretrained[word]
        else:
            print(word)
            return np.random.normal(0, 1, emb_dim)

    for word, i in vocabs.items():
        emb[i] = word_to_vec(word)
    return emb


def print_result(output, id2word, max_len):
    for sent in output:
        words = [id2word[id] for id in sent[:max_len]]
        print(" ".join(words))


def store_output(output, id2word, data_path, max_len):
    print("------------------ STORING OUTPUT -----------------")
    print("len of output: %d" % len(output))
    with open(data_path, 'wb') as fout:
        for sent in output:
            words = [id2word[id] for id in sent[:max_len]]
            rst = " ".join(words) + "\n"
            fout.write(rst.encode('utf-8'))


def pad_to_length(content, max_len, pad, bos=None, eos=None):
    """
    Pad sentence with <BOS>, <EOS>, <PAD>.
    if self.max_len = 3,
        if bos = <BOS>, then
            (a, b) -> (<BOS>, a, b, <EOS>, <PAD>)
        if bos is None, then
            (a, b) -> (a, b, <EOS>, <PAD>)
    :param content:
    :param max_len:
    :param eos:
    :param pad:
    :param bos:
    :return:
    """
    is_np_array = False
    if isinstance(content, np.ndarray):
        content = content.tolist()
        is_np_array = True
    rst = content + [eos] + [pad] * max_len
    if bos is not None:
        rst = [bos] + rst
        rtn = rst[: max_len + 2]
    else:
        rtn = rst[: max_len + 1]
    if is_np_array:
        return np.array(rtn)
    return rtn


def sent_to_ids(words, word2id, unk_id):
    """

    :param words:
    :param word2id:
    :param unk_id:
    :return:
    """
    return [word2id.get(word, unk_id) for word in words]
