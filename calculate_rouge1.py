import re
from collections import Counter
import string


def tokenize(text):
    tokens = []
    word = []
    for char in text:
        if char in string.whitespace or char in string.punctuation:
            if word:
                tokens.append(''.join(word))
                word = []
            tokens.append(char)
        else:
            word.append(char)
    if word:
        tokens.append(''.join(word))
    return tokens


def cal_rouge1(reference, candidate):
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)

    reference_counter = Counter(reference_tokens)
    candidate_counter = Counter(candidate_tokens)

    common = reference_counter & candidate_counter
    common_count = sum(common.values())

    recall = common_count / len(reference_tokens) if reference_tokens else 0
    precision = common_count / len(candidate_tokens) if candidate_tokens else 0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) else 0

    return recall



