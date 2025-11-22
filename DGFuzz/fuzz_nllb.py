"""Fuzzing script for the NLLB model to detect omission errors in translation."""

import os
import random
import string

# Disable parallelism warning for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import BertTokenizer as bt, BertForMaskedLM as bm
import jieba
import torch
import json
import argparse
from word_align import word_align
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# English stop words used when checking unaligned words
stop_words = set(stopwords.words('english'))

from Fuzz4SeMiss.calculate_rough1 import rouge_1_without_punctuation, rouge_1_with_punctuation
from word_align.awesome_align import modeling
from word_align.awesome_align.configuration_bert import BertConfig
from word_align.awesome_align.modeling import BertForMaskedLM
from word_align.awesome_align.tokenization_bert import BertTokenizer
from transformers import M2M100ForConditionalGeneration, NllbTokenizer
from Fuzz4SeMiss.eos_mutation import mutate_word_character, mutate_punctuations
from Fuzz4SeMiss.bert_mutation import predict_masked_word_by_position
from word_align.data_tokenize import transform
from word_align.word_align import set_seed


class Args:
    """Simple container for awesome-align configuration and paths."""

    def __init__(self):
        # Input sentence pair file used by awesome-align
        self.data_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input"
        # Alignment output file
        self.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/output.txt"
        # Pretrained awesome-align model path
        self.model_name_or_path = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        self.config_name = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        self.tokenizer_name = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/model_without_co"
        self.align_layer = 8
        self.extraction = 'softmax'
        self.softmax_threshold = 0.001
        self.output_prob_file = None
        self.output_word_file = None
        self.batch_size = 32
        self.cache_dir = None
        self.no_cuda = False
        self.num_workers = 4


def nllb_translate_with_token_and_eos_prob(model, tokenizer, input_text, device, beam_size=5):
    """
    Translate input_text with the NLLB model and record token / EOS probabilities
    for each generation step (best beam only).

    Returns:
        decoded_text (str): translated sentence
        chosen_beam_tokens_info (list): per-step info:
            (token_id, decoded_token, token_prob, eos_prob, token_prob - eos_prob)
    """
    # Tokenize and move to target device
    modle_inputs = tokenizer(input_text, return_tensors="pt").to(torch.device(device))

    # Generate translation with beam search and output scores for each step
    outputs = model.generate(
        **modle_inputs,
        num_beams=beam_size,
        output_scores=True,
        return_dict_in_generate=True,
        forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"],  # target language: Simplified Chinese
    )

    # Decode beam outputs
    decoded_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    beam_scores = outputs.scores  # list of logits for each generation step

    # Convert logits to probabilities
    probabilities = [torch.softmax(scores, dim=-1) for scores in beam_scores]
    eos_token_id = tokenizer.eos_token_id
    chosen_beam_tokens_info = []

    # Take the first (best) beam
    chosen_beam = outputs["sequences"][0]

    # Iterating over generated tokens (skip first one because it's usually BOS)
    for i in range(len(chosen_beam)):
        if i == 0:
            continue
        step_probs = probabilities[i - 1]
        chosen_token_index = chosen_beam[i].item()
        chosen_token_prob = step_probs[0, chosen_token_index].item()
        eos_prob = step_probs[0, eos_token_id].item()

        token_minus_eos_prob = chosen_token_prob - eos_prob

        decoded_token = tokenizer.decode(chosen_token_index)
        chosen_beam_tokens_info.append(
            (chosen_token_index, decoded_token, chosen_token_prob, eos_prob, token_minus_eos_prob)
        )

    print(chosen_beam_tokens_info)
    return decoded_texts[0], chosen_beam_tokens_info


def get_files_info(folder_path):
    """
    Get (file_name, full_path) for all files under folder_path, sorted by file name.
    """
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            files_info.append((file, full_path))
    files_info = sorted(files_info, key=lambda x: x[0])
    return files_info


# Process input text and run word alignment
def word_alignment(mutated_translated_text, mutated_text, args, word_align_model,
                   word_align_tokenizer, word_align_device):
    """
    Run awesome-align on (English source, translated target) and
    return the list of unaligned English words.
    """
    # Transform to awesome-align tokenized form
    awesome_tranlated_token, awesome_en_token = transform(mutated_translated_text, mutated_text)
    align_text = awesome_en_token + " ||| " + awesome_tranlated_token

    # Write single sentence pair to file for awesome-align
    input = open("/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input", mode='w', encoding='utf-8')
    input.write(align_text)
    input.close()

    # Set specific output file for omission detection
    args.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/detect_omission.txt"
    word_align.word_align(args, word_align_model, word_align_tokenizer, word_align_device)

    # Read first (and only) alignment line
    output_alignment = open(args.output_file, mode='r', encoding='utf-8').readline()
    print("output_alignment:")
    print(align_text)
    print(output_alignment)

    # Extract unaligned English words
    mutated_unaligned_word = get_unaligned_non_stop_words(
        awesome_en_token,
        awesome_tranlated_token,
        output_alignment
    )
    return mutated_unaligned_word


def fuzz_omission_nllb(args, nllb_model, nllb_tokenizer, nllb_device, word_align_model,
                       word_align_tokenizer, word_align_device, mask_model, mask_tokenizer, mask_device):
    """
    Main fuzzing procedure for NLLB.

    Two phases for each file:
    1) Guided fuzzing: use token_prob - eos_prob to guide which mutants to expand.
    2) No-guidance fuzzing: generate the same number of mutants per seed without guidance.

    Both phases:
      - Mutate English sentence using three operators (MO1, MO2, MO3).
      - Filter by ROUGE-1 similarity.
      - Translate with NLLB.
      - Run word alignment to detect omission-like cases (many unaligned words).
    """
    # Global counters across all files (8 datasets in the original paper)
    all_8_bug_num = 0
    all_8_bug_num_no_guidance = 0
    all_8_gen_num = 0
    all_8_gen_num_no_guidance = 0
    all_8_cov = 0
    all_8_cov_no_guidance = 0
    all_8_record_results = []
    all_8_record_results_no_guidance = []

    # Result paths
    result_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/results_nllb_wmt"
    gen_num_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/gen_num_result_nllb_wmt"
    file_path = "/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/omission_data_wmt"

    # Traverse all test files
    file_infos = get_files_info(file_path)
    for file_info in file_infos:
        test_data_path = file_info[1]
        file_name = file_info[0]
        print("start")

        # Paths to save generated tests and bug-triggering cases
        buglist_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/nllb/" + str(
            file_name) + "/buglist_nllb_allMRwmt_" + str(file_name)
        gen_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/nllb/" + str(
            file_name) + "/gen_data_nllb_allMRwmt" + str(file_name)
        buglist_no_guidance_data_path = buglist_data_path + "_no_guidance"
        gen_data_no_guidance_path = gen_data_path + "_no_guidance"

        # Load source sentences (one per line)
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines = test_data

        bug_list = []
        buglist_no_guidance = []
        single_gen_num = []  # number of generated mutants per seed
        bug_num = 0
        bug_num_no_guidance = 0
        total_count = 0
        total_count_no_guidance = 0
        gen_test = []  # all generated sentences in guided phase
        gen_test_no_gudiacnce = []  # all generated sentences in no-guidance phase
        max_trails = 50  # max times to expand one seed
        cov = 0  # number of seeds that revealed a bug
        cov_no_guidance = 0
        record_results = []  # (generated_tests, bugs) for plotting
        record_results_no_guidance = []
        m_num = 5  # number of mutations per popped seed
        minimum_prob = 1  # guidance threshold (min token_prob - eos_prob)

        # ------------------ Guided fuzzing phase ------------------
        while len(lines) > 0:
            sequence = []  # queue of sentences to mutate
            origianl_seed = lines.pop(0).strip()

            # Translate original seed and compute token_prob - eos_prob for guidance
            original_translated_text, original_chosen_beam_tokens_info = nllb_translate_with_token_and_eos_prob(
                nllb_model, nllb_tokenizer, origianl_seed, device=nllb_device
            )

            # Skip last 20% tokens (usually end of sentence)
            original_chosen_beam_tokens_info = original_chosen_beam_tokens_info[
                                               :int(-(len(original_chosen_beam_tokens_info) * 0.2))]

            diffs = [info[4] for info in original_chosen_beam_tokens_info]

            # Initialize minimum guidance value
            if len(diffs) == 0:
                pass
            else:
                minimum_prob = min(diffs)

            sequence.append(origianl_seed)
            single_count = 0  # how many mutants generated from this seed (guided phase)
            trails = 0
            flag = False  # whether this seed has triggered at least one bug

            # Pop sentences from queue and mutate them
            while len(sequence) > 0 and trails < max_trails:
                english_text = sequence.pop(0)
                words = english_text.split()

                for i in range(m_num):
                    if trails >= max_trails:
                        break
                    trails += 1

                    # Randomly choose mutation operator:
                    # 1: character mutation, 2: BERT MLM, 3: punctuation mutation
                    choice = random.randint(1, 3)
                    rand = random.randint(0, len(words) - 1)

                    if choice == 1:
                        mutated_text = mutate_word_character(english_text, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(
                            english_text, rand, mask_model, mask_tokenizer, mask_device
                        )
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)

                    # Skip if already generated before (avoid duplicates)
                    if mutated_text in gen_test:
                        continue

                    # Semantic similarity filter using ROUGE-1 with punctuation
                    rouge1 = rouge_1_with_punctuation(origianl_seed, mutated_text)
                    print("rouge1:  " + str(rouge1))
                    if rouge1 < 0.9:
                        continue

                    gen_test.append(mutated_text)
                    single_count += 1
                    total_count += 1
                    all_8_gen_num += 1

                    # Translate mutated sentence and compute guidance value again
                    mutated_translated_text, mutated_chosen_beam_tokens_info = nllb_translate_with_token_and_eos_prob(
                        nllb_model,
                        nllb_tokenizer,
                        mutated_text,
                        device=nllb_device
                    )
                    mutated_chosen_beam_tokens_info = mutated_chosen_beam_tokens_info[
                                                      :int(-(len(mutated_chosen_beam_tokens_info) * 0.2))]
                    diffs = [info[4] for info in mutated_chosen_beam_tokens_info]
                    if len(diffs) == 0:
                        pass
                    else:
                        mutated_min_diff = min(diffs)

                    # Word alignment to detect omission
                    mutated_unaligned_word = word_alignment(
                        mutated_translated_text,
                        mutated_text,
                        args,
                        word_align_model,
                        word_align_tokenizer,
                        word_align_device
                    )

                    # If too many unaligned English words -> treat as bug
                    if len(mutated_unaligned_word) > 3:
                        flag = True
                        bug_num += 1
                        all_8_bug_num += 1

                        bug_list.append(mutated_text)
                        bug_list_data = open(buglist_data_path, mode='a', encoding='utf-8')
                        if '\n' not in mutated_text:
                            mutated_text = mutated_text + '\n'
                        if '\n' not in mutated_translated_text:
                            mutated_translated_text = mutated_translated_text + '\n'
                        bug_list_data.write(mutated_text + mutated_translated_text)
                        bug_list_data.close()
                    else:
                        # Guidance: if mutated sentence has smaller token-eos diff,
                        # put it back into queue for further mutation
                        if mutated_min_diff < minimum_prob:
                            minimum_prob = mutated_min_diff
                            sequence.append(mutated_text)
                            print(minimum_prob)

                    # Record bug discovery curve per 10 generated tests
                    if total_count % 10 == 0:
                        record_results.append([total_count, bug_num])
                    if all_8_gen_num % 10 == 0:
                        all_8_record_results.append([all_8_gen_num, all_8_bug_num])

            # Coverage: this seed revealed at least one bug
            if flag:
                cov += 1
                all_8_cov += 1
            single_gen_num.append(single_count)

        # Save per-seed generation counts for the no-guidance phase
        write_gen_num = open(gen_num_path, mode='a', encoding='utf-8')
        write_gen_num_text = str(file_name) + ":" + str(single_gen_num) + "\n"
        write_gen_num.write(write_gen_num_text)
        write_gen_num.close()

        # ------------------ No-guidance fuzzing phase ------------------
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines2 = test_data

        while len(lines2) > 0:
            num = single_gen_num.pop(0)  # number of mutants to generate for this seed
            sequence = []
            origianl_seed = lines2.pop(0).strip()
            sequence.append(origianl_seed)
            count = 0
            flag_no_guidance = False  # coverage flag for this seed

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
                            english_text, rand, mask_model, mask_tokenizer, mask_device
                        )
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)

                    # Skip duplicates
                    if mutated_text in gen_test_no_gudiacnce:
                        continue

                    rouge1 = rouge_1_with_punctuation(origianl_seed, mutated_text)
                    print("rouge1:  " + str(rouge1))
                    if rouge1 < 0.9:
                        continue

                    count += 1
                    total_count_no_guidance += 1
                    all_8_gen_num_no_guidance += 1
                    gen_test_no_gudiacnce.append(mutated_text)

                    mutated_translated_text, mutated_chosen_beam_tokens_info = nllb_translate_with_token_and_eos_prob(
                        nllb_model,
                        nllb_tokenizer,
                        mutated_text,
                        device=nllb_device
                    )

                    # Word alignment without using guidance
                    mutated_unaligned_word = word_alignment(
                        mutated_translated_text,
                        mutated_text,
                        args,
                        word_align_model,
                        word_align_tokenizer,
                        word_align_device
                    )

                    if len(mutated_unaligned_word) > 3:
                        # Found a bug in no-guidance setting
                        bug_num_no_guidance += 1
                        all_8_bug_num_no_guidance += 1
                        flag_no_guidance = True
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
                        # Still push back into queue so we can continue mutating it
                        sequence.append(mutated_text)

                    # Record bug discovery curve per 10 generated tests
                    if total_count_no_guidance % 10 == 0:
                        record_results_no_guidance.append([total_count_no_guidance, bug_num_no_guidance])
                    if all_8_gen_num_no_guidance % 10 == 0:
                        all_8_record_results_no_guidance.append(
                            [all_8_gen_num_no_guidance, all_8_bug_num_no_guidance]
                        )

            # Coverage for this seed in no-guidance setting
            if flag_no_guidance:
                cov_no_guidance += 1
                all_8_cov_no_guidance += 1

        # ------------------ Save all generated tests and statistics ------------------
        write_guidance_all = open(gen_data_path, mode='a', encoding='utf-8')
        for item in gen_test:
            if '\n' not in item:
                item = item + '\n'
            write_guidance_all.write(item)
        write_guidance_all.close()

        write_non_guidance_all = open(gen_data_no_guidance_path, mode='a', encoding='utf-8')
        for item in gen_test_no_gudiacnce:
            if '\n' not in item:
                item = item + '\n'
            write_non_guidance_all.write(item)
        write_non_guidance_all.close()

        write_results = open(result_path, mode='a', encoding='utf-8')
        write_results_text = (
                str(buglist_data_path) + "\n" +
                "bug_num:  " + str(bug_num) + "\tgen_num:  " + str(len(gen_test)) +
                "\tcov:  " + str(cov) + "\n" + str(record_results) + "\n\n" +
                "bug_num_no_guidance:  " + str(bug_num_no_guidance) +
                "\tgen_num_no_guidance:  " + str(len(gen_test_no_gudiacnce)) +
                "\tcov_no_guidance:  " + str(cov_no_guidance) + "\n" +
                str(record_results_no_guidance) + "\n" + str(write_gen_num) + "\n\n\n\n"
        )
        write_results.write(write_results_text)
        write_results.close()

    # Global summary across all files
    all_8_record_results.append([all_8_gen_num, all_8_bug_num])
    all_8_record_results_no_guidance.append([all_8_gen_num_no_guidance, all_8_bug_num_no_guidance])
    all_8_write_results = open(result_path, mode='a', encoding='utf-8')
    all_8_write_results_text = (
            "all_8:" + "\n" +
            "all_8_bug_num:  " + str(all_8_bug_num) +
            "\tall_8_gen_num:  " + str(all_8_gen_num) +
            "\tall_8_cov:  " + str(all_8_cov) + "\n" +
            str(all_8_record_results) + "\n\n" +
            "all8_no_guidance_bug_num:  " + str(all_8_bug_num_no_guidance) +
            "\tgen_num_no_guidance:  " + str(all_8_gen_num_no_guidance) +
            "\tcov_no_guidance:  " + str(all_8_cov_no_guidance) + "\n" +
            str(all_8_record_results_no_guidance) + "\n\n\n\n"
    )
    all_8_write_results.write(all_8_write_results_text)
    all_8_write_results.close()


def get_unaligned_non_stop_words(english_sentence, translated_sentence, alignment_sequence):
    """
    From an alignment string "i-j ..." return English words whose indices i
    do not appear in the alignment and are not stop words or punctuation.
    """
    english_tokens = english_sentence.split()
    translated_tokens = translated_sentence.split()

    # Extract aligned English token indices from alignment string like "0-0 1-2 ..."
    aligned_indices = set(int(pair.split('-')[0]) for pair in alignment_sequence.split())

    word_index_dict = {
        index: word
        for index, word in enumerate(english_tokens)
        if word.lower() not in stop_words and word not in string.punctuation
    }

    # Words whose indices are never aligned are considered "unaligned"
    unaligned_words = [word for index, word in word_index_dict.items() if index not in aligned_indices]

    return unaligned_words


def get_files_info(folder_path):
    """
    Get (file_name, full_path) for all files under folder_path.

    Note: this definition overrides the previous sorted version.
    """
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            files_info.append((file, full_path))
    return files_info


if __name__ == '__main__':
    # Prepare args and devices
    args = Args()
    nllb_device = "cuda:1"
    word_align_device = "cuda:2"
    mask_device = "cuda:3"

    # Set random seed for reproducibility
    set_seed(args)

    # ------------ Load awesome-align model ------------
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
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it "
            "from another script, save it, and load it from here, using --tokenizer_name".format(
                tokenizer_class.__name__
            )
        )

    # Set special token IDs used by awesome-align
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

    # ------------ Load NLLB translation model ------------
    nllb_model_path = '/home/ubuntu/Desktop/DFuzz4NMT/translator/model/nllb-1.3b'
    nllb_model = M2M100ForConditionalGeneration.from_pretrained(nllb_model_path).to(torch.device("cuda:1"))
    nllb_tokenizer = NllbTokenizer.from_pretrained(nllb_model_path)

    # ------------ Load BERT masked LM for mutation operator ------------
    mask_model_path = "/home/ubuntu/Desktop/bert-large-uncased"
    mask_tokenizer = bt.from_pretrained(mask_model_path)
    mask_model = bm.from_pretrained(mask_model_path).to(torch.device("cuda:3"))

    # ------------ Run fuzzing ------------
    fuzz_omission_nllb(
        args,
        nllb_model,
        nllb_tokenizer,
        nllb_device,
        word_align_model,
        word_align_tokenizer,
        word_align_device,
        mask_model,
        mask_tokenizer,
        mask_device
    )
