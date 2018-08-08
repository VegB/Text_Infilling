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


def calculate_bleu(reference_path, candidate_path):
    reference = get_reference(reference_path)
    candidate = get_candidate(candidate_path)
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
    bleu2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=SmoothingFunction().method4)
    bleu3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=SmoothingFunction().method4)
    bleu4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=SmoothingFunction().method4)
    return [bleu1, bleu2, bleu3, bleu4]
