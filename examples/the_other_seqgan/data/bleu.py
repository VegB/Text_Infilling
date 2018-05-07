"""
How to run:
python bleu.py [directory of generated files] [path to coco.txt]
"""
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_candidate(file_path):
    with open(file_path, 'r') as fin:
        data = fin.readlines()
        candidate = []
        for line in data:
            candidate.extend(line.strip().split())
    return candidate


def get_reference(file_path):
    with open(file_path, 'r') as fin:
        data = fin.readlines()
    reference = []
    for line in data:
        reference.extend(line.strip().split())
    reference = [reference]
    return reference


_, dir_path, candidate_path = sys.argv

candidate = get_candidate(candidate_path)
# candidate = get_candidate('coco.txt')
# dir_path = 'texy_coco/'
fout = open(dir_path + 'bleu_out.txt', 'a+')
for i in range(19):
    prefix = dir_path + '/'
    ref_name = prefix + str(i * 10) + '.txt'
    reference = get_reference(ref_name)
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function = SmoothingFunction().method4)
    bleu2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function = SmoothingFunction().method4)
    bleu3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function = SmoothingFunction().method4)
    bleu4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function = SmoothingFunction().method4)
    buf = ref_name + '\n%f\n%f\n%f\n%f\n\n' % (bleu1, bleu2, bleu3, bleu4)
    print(buf)
    fout.write(buf)
fout.close()
