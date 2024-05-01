import random
import string
import torch
def mutate_punctuations(text):
    punctuations=string.punctuation
    replacement_punctuations = punctuations
    punctuation_indices = [i for i, char in enumerate(text) if char in punctuations]
    if not punctuation_indices:
        return text
    index = random.choice(punctuation_indices)
    mutation_type = random.randint(1, 4)
    mutated_text = text[:index]

    if mutation_type == 1:
        mutated_text += random.choice(replacement_punctuations)
    elif mutation_type == 2:
        if random.choice([True, False]):
            if index != 0 and text[index - 1] != ' ':
                mutated_text += " " + text[index]
            else:
                mutated_text += text[index]
        else:
            mutated_text += text[index]
            if (index + 1) < len(text) and text[index + 1] != ' ':
                mutated_text += " "
    elif mutation_type == 3:
        if random.choice([True, False]):
            mutated_text += random.choice(punctuations) + text[index]
        else:
            mutated_text += text[index] + random.choice(punctuations)
    elif mutation_type == 4:
        pass

    mutated_text += text[index + 1:]

    return mutated_text




def mutate_word(word):
    characters = string.ascii_letters

    if not word:
        return word

    mutation_type = random.randint(1, 3)
    mutation_index = random.randint(0, len(word) - 1)

    if mutation_type == 1:
        mutated_word = word[:mutation_index] + random.choice(characters) + word[mutation_index + 1:]
    elif mutation_type == 2:
        mutated_word = word[:mutation_index] + random.choice(characters) + word[mutation_index:]
    elif mutation_type == 3:
        if len(word) > 1:
            mutated_word = word[:mutation_index] + word[mutation_index + 1:]
        else:
            mutated_word = ""

    return mutated_word

def mutate_word_character(sentence, word_index):

    words = sentence.split()
    if word_index >= len(words) or word_index < 0:
        return sentence

    word = words[word_index]

    if word[-1] in string.punctuation:

        mutated_word = mutate_word(word[:-1])
        mutated_word += word[-1]
    else:
        mutated_word = mutate_word(word)

    words[word_index] = mutated_word

    return ' '.join(words)


def predict_masked_word_by_position(sentence, word_index,model,tokenizer,device):

    words = sentence.split()
    if word_index >= len(words) or word_index < 0:
        return sentence
    if words[word_index][-1] in string.punctuation:
        masked_word = words[word_index][:-1]
        punctuation = words[word_index][-1]
    else:
        masked_word = words[word_index]
        punctuation = ''

    words[word_index] = tokenizer.mask_token + punctuation
    masked_sentence = ' '.join(words)
    inputs = tokenizer(masked_sentence, return_tensors='pt').to(torch.device(device))
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits

    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

    predicted_token_ids = predictions[0, mask_token_index, :].topk(10).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token_id]).strip() for token_id in predicted_token_ids]
    for token in predicted_tokens[:]:
        if not all(char.isalpha() for char in token):
            predicted_tokens.remove(token)

    if len(predicted_tokens)==0:
        return sentence
    else:
        chosen_word = random.choice(predicted_tokens)

    words[word_index] = chosen_word + punctuation
    return ' '.join(words)



