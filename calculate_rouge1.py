import re
from collections import Counter
import string


# Split text into a list of tokens (words + punctuation + whitespace as separate pieces)
def tokenize(text):
    tokens = []
    word = []
    for char in text:
        # If current char is whitespace or punctuation, end current word
        if char in string.whitespace or char in string.punctuation:
            if word:
                tokens.append(''.join(word))
                word = []
            # Keep the punctuation/whitespace as its own token
            tokens.append(char)
        else:
            # Build up a word
            word.append(char)
    # Add the last word if there is one
    if word:
        tokens.append(''.join(word))
    return tokens


# Calculate ROUGE-1 recall between a reference sentence and a candidate sentence
def cal_rouge1(reference, candidate):
    # Tokenize both sentences
    reference_tokens = tokenize(reference)
    candidate_tokens = tokenize(candidate)

    # Count token frequencies
    reference_counter = Counter(reference_tokens)
    candidate_counter = Counter(candidate_tokens)

    # Multiset intersection -> overlapping tokens
    common = reference_counter & candidate_counter
    common_count = sum(common.values())

    # Compute recall, precision and F1 (but only return recall)
    recall = common_count / len(reference_tokens) if reference_tokens else 0
    precision = common_count / len(candidate_tokens) if candidate_tokens else 0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) else 0

    return recall
