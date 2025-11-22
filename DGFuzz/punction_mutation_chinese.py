import random
import string

from zhon.hanzi import punctuation as ch_punctuation


def mutate_chinese_punctuations(text):

    english_punctuations = string.punctuation
    chinese_punctuations = ch_punctuation
    punctuations = english_punctuations + chinese_punctuations

    punctuation_indices = [i for i, char in enumerate(text) if char in punctuations]

    if not punctuation_indices:
        return text

    index = random.choice(punctuation_indices)
    mutation_type = random.randint(1, 4)
    mutated_text = text[:index]

    if mutation_type == 1:
        mutated_text += random.choice(punctuations)
    elif mutation_type == 2:
        if random.choice([True, False]):
            if index > 0 and text[index - 1] != ' ':
                mutated_text += ' ' + text[index]
            else:
                mutated_text += text[index]
        else:
            mutated_text += text[index]
            if index + 1 < len(text) and text[index + 1] != ' ':
                mutated_text += ' '
    elif mutation_type == 3:
        if random.choice([True, False]):
            mutated_text += random.choice(punctuations) + text[index]
        else:
            mutated_text += text[index] + random.choice(punctuations)


    mutated_text += text[index + 1:]

    return mutated_text
