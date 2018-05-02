with open('new.txt', 'r') as fin:
    data = fin.readlines()

test, oracle = [], []
for line in data[::2]:
    test.append(line.strip().split()[-1])
for line in data[1::2]:
    oracle.append(line.strip().split()[-1])

with open('nll_test.txt', 'w') as fout:
    fout.write('\n'.join(test))
with open('nll_oracle.txt' , 'w') as fout:
    fout.write('\n'.join(oracle))

