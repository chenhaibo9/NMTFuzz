import random
import string
import torch


# Randomly modify one punctuation character in the text
def mutate_punctuations(text):
    punctuations = string.punctuation
    replacement_punctuations = punctuations

    # Find all indices of punctuation characters
    punctuation_indices = [i for i, char in enumerate(text) if char in punctuations]
    if not punctuation_indices:
        return text

    # Randomly pick one punctuation position
    index = random.choice(punctuation_indices)
    # Randomly choose mutation type 1–4
    mutation_type = random.randint(1, 4)
    mutated_text = text[:index]

    if mutation_type == 1:
        # Replace with another random punctuation
        mutated_text += random.choice(replacement_punctuations)
    elif mutation_type == 2:
        # Change spaces around the punctuation
        if random.choice([True, False]):
            #  add a space before
            if index != 0 and text[index - 1] != ' ':
                mutated_text += " " + text[index]
            else:
                mutated_text += text[index]
        else:
            #  add a space after
            mutated_text += text[index]
            if (index + 1) < len(text) and text[index + 1] != ' ':
                mutated_text += " "
    elif mutation_type == 3:
        # Duplicate punctuation or add another near it
        if random.choice([True, False]):
            mutated_text += random.choice(punctuations) + text[index]
        else:
            mutated_text += text[index] + random.choice(punctuations)


    # Add the rest of the text
    mutated_text += text[index + 1:]

    return mutated_text


# Randomly mutate a single word at character level
def mutate_word(word):
    characters = string.ascii_letters

    if not word:
        return word

    # Choose mutation type and position
    mutation_type = random.randint(1, 3)
    mutation_index = random.randint(0, len(word) - 1)

    if mutation_type == 1:
        # Replace one character
        mutated_word = word[:mutation_index] + random.choice(characters) + word[mutation_index + 1:]
    elif mutation_type == 2:
        # Insert a new character
        mutated_word = word[:mutation_index] + random.choice(characters) + word[mutation_index:]
    elif mutation_type == 3:
        # Delete one character (if length > 1)
        if len(word) > 1:
            mutated_word = word[:mutation_index] + word[mutation_index + 1:]
        else:
            mutated_word = ""

    return mutated_word


# Mutate one word in a sentence by index
def mutate_word_character(sentence, word_index):
    words = sentence.split()
    # If index is invalid, return original sentence
    if word_index >= len(words) or word_index < 0:
        return sentence

    word = words[word_index]

    # Keep punctuation at the end of the word unchanged
    if word[-1] in string.punctuation:
        mutated_word = mutate_word(word[:-1])
        mutated_word += word[-1]
    else:
        mutated_word = mutate_word(word)

    words[word_index] = mutated_word

    return ' '.join(words)


# Use a masked language model to replace a word at a given position
def predict_masked_word_by_position(sentence, word_index, model, tokenizer, device):
    words = sentence.split()
    # If index is invalid, return original sentence
    if word_index >= len(words) or word_index < 0:
        return sentence

    # Separate word and trailing punctuation
    if words[word_index][-1] in string.punctuation:
        masked_word = words[word_index][:-1]
        punctuation = words[word_index][-1]
    else:
        masked_word = words[word_index]
        punctuation = ''

    # Replace the word with the [MASK] token (keep punctuation)
    words[word_index] = tokenizer.mask_token + punctuation
    masked_sentence = ' '.join(words)

    # Run masked LM to get predictions
    inputs = tokenizer(masked_sentence, return_tensors='pt').to(torch.device(device))
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits

    # Find the position of the [MASK] token
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

    # Get top 10 predicted token ids for the mask position
    predicted_token_ids = predictions[0, mask_token_index, :].topk(10).indices[0].tolist()
    predicted_tokens = [tokenizer.decode([token_id]).strip() for token_id in predicted_token_ids]

    # Remove predictions that contain non-letter characters
    for token in predicted_tokens[:]:
        if not all(char.isalpha() for char in token):
            predicted_tokens.remove(token)

    # If no valid prediction, keep original sentence
    if len(predicted_tokens) == 0:
        return sentence
    else:
        # Randomly choose one predicted word
        chosen_word = random.choice(predicted_tokens)

    # Put chosen word back with original punctuation
    words[word_index] = chosen_word + punctuation
    return ' '.join(words)
