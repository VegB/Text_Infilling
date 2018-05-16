with open('log.txt', 'r') as fin:
    data = fin.readlines()
    data = [line.split(',') for line in data]

train_ppl, valid_ppl, test_ppl = [], [], []

for para in data:
    para = [line.split() for line in para]
    train_ppl.append(para[1][-1])
    valid_ppl.append(para[2][-1])
    test_ppl.append(para[3][-1])

with open('train_ppl.txt', 'w') as fout:
    fout.write('\n'.join(train_ppl))
with open('valid_ppl.txt', 'w') as fout:
    fout.write('\n'.join(valid_ppl))
with open('test_ppl.txt', 'w') as fout:
    fout.write('\n'.join(test_ppl))

