def split_syn():
    with open("syn.txt", "r") as fin:
        data = fin.readlines()
    with open("syn_test_nll.txt", "w") as fout:
        rst = []
        for line in data[::2]:
            line = line.split()
            rst.append(line[-1])
        fout.write('\n'.join(rst))
    with open("syn_oracle_nll.txt", "w") as fout:
        rst = []
        for line in data[1::2]:
            line = line.split()
            rst.append(line[-1])
        fout.write('\n'.join(rst))


def write_sh():
    with open("seqgan.sh", "w") as fout:
        for i in range(19):
            rst = "python ../bleu.py eval_full/%d.txt coco.txt\n" % (10 * i)
            fout.write(rst)
        for i in range(19):
            rst = "python ../bleu.py eval_full/%d.eval.txt coco.eval.txt\n" % (10 * i)
            fout.write(rst)


def split_bleu(type):
    bleu = [[], [], [], []]
    with open(type + ".bleu.txt", "r") as fin:
        data = fin.readlines()
        data = [line.split() for line in data]
        for line in data:
            for i in range(4):
                bleu[i].append(line[i])
    for i in range(4):
        with open(type + ".bleu-%d.txt" % (i + 1), "w") as fout:
            fout.write('\n'.join(bleu[i]))

split_bleu(type="eval")
