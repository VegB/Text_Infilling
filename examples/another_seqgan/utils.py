import numpy as np


def print_and_write_to_file(rst, fout, print_out=True):
    if print_out:
        print(rst)
    fout.write(rst)
    fout.flush()


def print_result(output, id2word, max_len):
    for sent in output:
        words = [id2word[id] for id in sent[:max_len]]
        print(" ".join(words).encode("utf-8"))


def store_output(output, id2word, data_path, max_len):
    print("------------------ STORING OUTPUT -----------------")
    print("len of output: %d" % len(output))
    with open(data_path, 'wb') as fout:
        for sent in output:
            words = [id2word[id] for id in sent[:max_len]]
            rst = " ".join(words) + "\n"
            fout.write(rst.encode('utf-8'))


def pad_to_length(content, max_len, pad):
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
    rst = content + [pad] * max_len
    rtn = rst[: max_len]
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


def split_nll(filepath, dstpath):
    with open(filepath, 'r') as fin:
        data = fin.readlines()
        data = [line.strip().split()[-1] for line in data]
    with open(dstpath, 'w') as fout:
        fout.write('oracle:\n' + '\n'.join(data[::2]) + '\n------------\ngen:\n')
        fout.write('\n'.join(data[1::2]))


# split_nll("./data/log.txt", "./data/nll.txt")
