import os
import random
import string

from transformers import BertTokenizer as bt, BertForMaskedLM as bm
import jieba
import torch
import json
import argparse
from word_align import word_align
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
from calculate_rouge1 import cal_rouge1
from word_align.awesome_align import modeling
from word_align.awesome_align.configuration_bert import BertConfig
from word_align.awesome_align.modeling import BertForMaskedLM
from word_align.awesome_align.tokenization_bert import BertTokenizer
from transformers import MarianMTModel, MarianTokenizer
from mutation import mutate_word_character, mutate_punctuations
from mutation import predict_masked_word_by_position
from word_align.data_tokenize import transform


# Simple container for awesome-align arguments and file paths
class Args:
    def __init__(self):
        # Input file for awesome-align
        self.data_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input"
        # Output alignment file for awesome-align
        self.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/output.txt"
        # Pretrained awesome-align model path
        self.model_name_or_path = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        self.config_name = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        self.tokenizer_name = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        # Alignment layer index
        self.align_layer = 8
        self.extraction = 'softmax'
        self.softmax_threshold = 0.001
        self.output_prob_file = None
        self.output_word_file = None
        self.batch_size = 32
        self.cache_dir = None
        self.no_cuda = False
        self.num_workers = 4


# Translate with Helsinki model and collect token / EOS probabilities of the best beam
def helsinki_translate_with_token_and_eos_prob(model, tokenizer, input_text, device, beam_size=5):
    # Encode source sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(torch.device(device))

    # Generate with beam search and keep scores
    outputs = model.generate(
        inputs,
        max_length=256,
        num_beams=beam_size,
        return_dict_in_generate=True,
        output_scores=True
    )
    decoded_texts = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
    beam_scores = outputs["scores"]

    # Softmax over vocab for each time step
    probabilities = [torch.softmax(scores, dim=-1) for scores in beam_scores]
    eos_token_id = tokenizer.eos_token_id

    chosen_beam_tokens_info = []
    # Take first beam as chosen one
    chosen_beam = outputs["sequences"][0]

    # Iterate over generated tokens
    for i in range(len(chosen_beam)):
        if i == 0:
            # Skip first token (usually BOS)
            continue
        step_probs = probabilities[i - 1]
        chosen_token_index = chosen_beam[i].item()
        chosen_token_prob = step_probs[0, chosen_token_index].item()
        eos_prob = step_probs[0, eos_token_id].item()

        # Difference between chosen token and EOS probability
        token_minus_eos_prob = chosen_token_prob - eos_prob

        decoded_token = tokenizer.decode(chosen_token_index)
        chosen_beam_tokens_info.append(
            (chosen_token_index, decoded_token, chosen_token_prob, eos_prob, token_minus_eos_prob)
        )

    # Return translation and per-token info list
    return decoded_texts[0], chosen_beam_tokens_info


# Get all files (name, full_path) in a folder, sorted by file name
def get_files_info(folder_path):
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            files_info.append((file, full_path))
    files_info = sorted(files_info, key=lambda x: x[0])
    return files_info


# Run word alignment and return list of unaligned English words
def word_alignment(mutated_translated_text, mutated_text, args, word_align_model,
                   word_align_tokenizer, word_align_device):
    # Convert to awesome-align tokenized format
    awesome_tranlated_token, awesome_en_token = transform(mutated_translated_text, mutated_text)
    align_text = awesome_en_token + " ||| " + awesome_tranlated_token

    # Write source ||| target to input file
    input = open("/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input", mode='w', encoding='utf-8')
    input.write(align_text)
    input.close()

    # Set output file for detection
    args.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/detect_omission.txt"
    # Run awesome-align
    word_align.word_align(args, word_align_model, word_align_tokenizer, word_align_device)

    # Read first alignment line
    output_alignment = open(args.output_file, mode='r', encoding='utf-8').readline()
    print("output_alignment:")
    print(align_text)
    print(output_alignment)

    # Get unaligned non-stopwords on English side
    mutated_unaligned_word = get_unaligned_non_stop_words(
        awesome_en_token,
        awesome_tranlated_token,
        output_alignment
    )
    return mutated_unaligned_word


# Main fuzzing loop for Helsinki model (guided + non-guided)
def fuzz_omission_helsinki(args, helsinki_model, helsinki_tokenizer, helsinki_device, word_align_model,
                           word_align_tokenizer, word_align_device, mask_model, mask_tokenizer, mask_device):
    # File to store number of generated tests per seed
    gen_num_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki_gen_num_result"
    # Directory of test datasets
    file_path = "/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/dataset"
    file_infos = get_files_info(file_path)

    # Loop over each dataset file
    for file_info in file_infos:
        test_data_path = file_info[1]
        file_name = file_info[0]

        # Output paths for bug lists and generated tests
        buglist_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki/" + str(
            file_name) + "/buglist_allMR_" + str(file_name)
        gen_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki/" + str(
            file_name) + "/gen_data_allMR_" + str(file_name)
        buglist_no_guidance_data_path = buglist_data_path + "_no_guidance"
        gen_data_no_guidance_path = gen_data_path + "_no_guidance"

        # Read original test sentences (one per line)
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines = test_data

        bug_list = []
        buglist_no_guidance = []
        single_gen_num = []          # number of mutants per seed (guided)
        gen_test = []                # all mutants in guided phase
        gen_test_no_gudiacnce = []   # all mutants in non-guided phase

        max_trails = 50              # max times to explore around one seed
        m_num = 5                    # number of mutations per sentence in one round
        minimum_prob = 1             # guidance threshold

        # -------- Guided fuzzing --------
        while len(lines) > 0:
            sequence = []
            # Pop one seed sentence
            origianl_seed = lines.pop(0).strip()
            print("Original Text:", origianl_seed)

            # Translate original seed and compute token-EOS diffs
            original_translated_text, original_chosen_beam_tokens_info = helsinki_translate_with_token_and_eos_prob(
                helsinki_model, helsinki_tokenizer, origianl_seed, device=helsinki_device)

            # Discard last 20% of tokens (end of sentence)
            original_chosen_beam_tokens_info = original_chosen_beam_tokens_info[
                                               :int(-(len(original_chosen_beam_tokens_info) * 0.2))]
            diffs = [info[4] for info in original_chosen_beam_tokens_info]
            if len(diffs) == 0:
                pass
            else:
                # Initial minimum diff for guidance
                minimum_prob = min(diffs)

            sequence.append(origianl_seed)
            single_count = 0          # mutants generated from this seed (guided)
            trails = 0               # number of fuzzing trials for this seed

            # BFS over mutated sentences with guidance
            while len(sequence) > 0 and trails < max_trails:
                english_text = sequence.pop(0)
                words = english_text.split()

                # Try up to m_num mutations around this sentence
                for i in range(m_num):
                    if trails >= max_trails:
                        break
                    trails += 1

                    # Randomly choose one mutation operator
                    choice = random.randint(1, 3)
                    rand = random.randint(0, len(words) - 1)
                    if choice == 1:
                        mutated_text = mutate_word_character(english_text, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(
                            english_text, rand, mask_model, mask_tokenizer, mask_device)
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)

                    # Skip duplicates
                    if mutated_text in gen_test:
                        continue

                    # Similarity filter using ROUGE-1
                    rouge1 = cal_rouge1(origianl_seed, mutated_text)
                    if rouge1 < 0.9:
                        continue

                    gen_test.append(mutated_text)
                    single_count += 1

                    # Translate mutant and compute token-EOS diffs
                    mutated_translated_text, mutated_chosen_beam_tokens_info = helsinki_translate_with_token_and_eos_prob(
                        helsinki_model,
                        helsinki_tokenizer,
                        mutated_text,
                        device=helsinki_device
                    )
                    mutated_chosen_beam_tokens_info = mutated_chosen_beam_tokens_info[
                                                      :int(-(len(mutated_chosen_beam_tokens_info) * 0.2))]
                    diffs = [info[4] for info in mutated_chosen_beam_tokens_info]
                    if len(diffs) == 0:
                        pass
                    else:
                        mutated_min_diff = min(diffs)

                    # Word alignment to detect omissions
                    mutated_unaligned_word = word_alignment(
                        mutated_translated_text,
                        mutated_text,
                        args,
                        word_align_model,
                        word_align_tokenizer,
                        word_align_device
                    )

                    # If many unaligned words, treat as bug
                    if len(mutated_unaligned_word) > 3:
                        print("append")
                        print(mutated_text)
                        print(mutated_translated_text)
                        bug_list.append(mutated_text)
                        bug_list_data = open(buglist_data_path, mode='a', encoding='utf-8')
                        if '\n' not in mutated_text:
                            mutated_text = mutated_text + '\n'
                        if '\n' not in mutated_translated_text:
                            mutated_translated_text = mutated_translated_text + '\n'
                        bug_list_data.write(mutated_text + mutated_translated_text)
                        bug_list_data.close()
                    else:
                        # Guidance: if diff is smaller, keep exploring this mutant
                        if mutated_min_diff < minimum_prob:
                            minimum_prob = mutated_min_diff
                            sequence.append(mutated_text)

            # Record number of generated tests for this seed
            single_gen_num.append(single_count)

        # Append generation numbers for this file to global record
        write_gen_num = open(gen_num_path, mode='a', encoding='utf-8')
        write_gen_num_text = str(file_name) + ":" + str(single_gen_num) + "\n"
        write_gen_num.write(write_gen_num_text)
        write_gen_num.close()

        # -------- Non-guided fuzzing (replay same counts) --------
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines2 = test_data

        # For each seed, generate the same number of mutants but without guidance
        while len(lines2) > 0:
            num = single_gen_num.pop(0)
            sequence = []
            origianl_seed = lines2.pop(0).strip()
            sequence.append(origianl_seed)
            count = 0
            while count < num:
                if len(sequence) == 0:
                    sequence.append(origianl_seed)
                english_text = sequence.pop(0).strip()
                words = english_text.split()

                for i in range(m_num):
                    if count >= num:
                        break
                    choice = random.randint(1, 3)
                    rand = random.randint(0, len(words) - 1)

                    if choice == 1:
                        mutated_text = mutate_word_character(english_text, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(
                            english_text, rand, mask_model, mask_tokenizer, mask_device)
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)

                    # Skip duplicates in non-guided list
                    if mutated_text in gen_test_no_gudiacnce:
                        continue

                    # Similarity filter
                    rouge1 = cal_rouge1(origianl_seed, mutated_text)
                    if rouge1 < 0.9:
                        continue

                    count += 1
                    gen_test_no_gudiacnce.append(mutated_text)

                    # Translate and align (no guidance here)
                    mutated_translated_text, mutated_chosen_beam_tokens_info = helsinki_translate_with_token_and_eos_prob(
                        helsinki_model,
                        helsinki_tokenizer,
                        mutated_text,
                        device=helsinki_device
                    )

                    mutated_unaligned_word = word_alignment(
                        mutated_translated_text,
                        mutated_text,
                        args,
                        word_align_model,
                        word_align_tokenizer,
                        word_align_device
                    )
                    print(mutated_unaligned_word)

                    # Save bug triggered in non-guided phase
                    if len(mutated_unaligned_word) > 3:
                        print("append")
                        print(mutated_text)
                        print(mutated_translated_text)
                        buglist_no_guidance.append(mutated_text)
                        buglist_no_guidance_data = open(buglist_no_guidance_data_path, mode='a', encoding='utf-8')
                        if '\n' not in mutated_text:
                            mutated_text = mutated_text + '\n'
                        if '\n' not in mutated_translated_text:
                            mutated_translated_text = mutated_translated_text + '\n'
                        buglist_no_guidance_data.write(mutated_text + mutated_translated_text)
                        buglist_no_guidance_data.close()
                    else:
                        sequence.append(mutated_text)

        # Write all guided mutants
        write_guidance_all = open(gen_data_path, mode='a', encoding='utf-8')
        for item in gen_test:
            if '\n' not in item:
                item = item + '\n'
            write_guidance_all.write(item)
        write_guidance_all.close()

        # Write all non-guided mutants
        write_non_guidance_all = open(gen_data_no_guidance_path, mode='a', encoding='utf-8')
        for item in gen_test_no_gudiacnce:
            if '\n' not in item:
                item = item + '\n'
            write_non_guidance_all.write(item)
        write_non_guidance_all.close()


# Get English words that are not aligned to any target word
def get_unaligned_non_stop_words(english_sentence, translated_sentence, alignment_sequence):
    english_tokens = english_sentence.split()

    # Indices on English side that are aligned
    aligned_indices = set(int(pair.split('-')[0]) for pair in alignment_sequence.split())

    # Filter tokens: not stopword and not punctuation
    word_index_dict = {
        index: word
        for index, word in enumerate(english_tokens)
        if word.lower() not in stop_words and word not in string.punctuation
    }

    # Words whose indices are not in aligned set
    unaligned_words = [word for index, word in word_index_dict.items() if index not in aligned_indices]

    return unaligned_words


# NOTE: this redefines get_files_info again (overrides the previous one)
def get_files_info(folder_path):
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            files_info.append((file, full_path))
    return files_info


if __name__ == '__main__':

    args = Args()
    # Set devices for different models
    helsinki_device = "cuda:1"
    word_align_device = "cuda:2"
    mask_device = "cuda:3"

    # Load awesome-align config, tokenizer and model
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        word_align_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        word_align_tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    # Set special token IDs for awesome-align
    modeling.PAD_ID = word_align_tokenizer.pad_token_id
    modeling.CLS_ID = word_align_tokenizer.cls_token_id
    modeling.SEP_ID = word_align_tokenizer.sep_token_id

    if args.model_name_or_path:
        word_align_model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = model_class(config=config)
    word_align_model.to(torch.device("cuda:2"))

    # Load Helsinki NMT model and tokenizer
    helsinki_model_path = '/home/ubuntu/Desktop/DFuzz4NMT/translator/model/helsinki'
    helsinki_model = MarianMTModel.from_pretrained(helsinki_model_path).to(torch.device("cuda:1"))
    helsinki_tokenizer = MarianTokenizer.from_pretrained(helsinki_model_path)

    # Load masked LM (BERT) for mutation operator 2
    mask_model_path = "/home/ubuntu/Desktop/bert-large-uncased"
    mask_tokenizer = bt.from_pretrained(mask_model_path)
    mask_model = bm.from_pretrained(mask_model_path).to(torch.device("cuda:3"))

    # Start fuzzing
    fuzz_omission_helsinki(
        args,
        helsinki_model,
        helsinki_tokenizer,
        helsinki_device,
        word_align_model,
        word_align_tokenizer,
        word_align_device,
        mask_model,
        mask_tokenizer,
        mask_device
    )
