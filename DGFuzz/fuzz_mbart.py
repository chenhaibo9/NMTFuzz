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
from word_align.awesome_align.tokenization_bert import BertTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from mutation import mutate_word_character, mutate_punctuations
from mutation import predict_masked_word_by_position
from word_align.data_tokenize import transform



class Args:
    def __init__(self):
        self.data_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input"
        self.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/output.txt"
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


def mbart_translate_with_token_and_eos_prob(model, tokenizer, input_text, device, beam_size=5):
    src_lang = "en_XX"
    tgt_lang = "zh_CN"
    model_inputs = tokenizer(input_text, return_tensors="pt").to(torch.device(device))
    outputs = model.generate(**model_inputs, num_beams=beam_size, output_scores=True, return_dict_in_generate=True,
                             forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    decoded_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    beam_scores = outputs.scores
    probabilities = [torch.softmax(scores, dim=-1) for scores in beam_scores]
    eos_token_id = tokenizer.eos_token_id
    chosen_beam_tokens_info = []
    chosen_beam = outputs["sequences"][0]

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
            (chosen_token_index, decoded_token, chosen_token_prob, eos_prob, token_minus_eos_prob))

    return decoded_texts[0], chosen_beam_tokens_info


def get_files_info(folder_path):
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            files_info.append((file, full_path))
    files_info = sorted(files_info, key=lambda x: x[0])
    return files_info



def word_alignment(mutated_translated_text, mutated_text, args, word_align_model,
                   word_align_tokenizer, word_align_device):
    awesome_tranlated_token, awesome_en_token = transform(mutated_translated_text, mutated_text)
    align_text = awesome_en_token + " ||| " + awesome_tranlated_token
    input = open("/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input", mode='w', encoding='utf-8')
    input.write(align_text)
    input.close()
    args.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/detect_omission.txt"
    word_align.word_align(args, word_align_model, word_align_tokenizer, word_align_device)
    output_alignment = open(args.output_file, mode='r', encoding='utf-8').readline()
    print("output_alignment:")
    print(align_text)
    print(output_alignment)
    mutated_unaligned_word = get_unaligned_non_stop_words(awesome_en_token, awesome_tranlated_token,
                                                          output_alignment)
    return mutated_unaligned_word


def fuzz_omission_mbart(args, mbart_model, mbart_tokenizer, mbart_device, word_align_model,
                           word_align_tokenizer, word_align_device, mask_model, mask_tokenizer, mask_device):
    gen_num_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/mbart_gen_num_result"
    file_path = "/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/dataset"
    file_infos = get_files_info(file_path)
    for file_info in file_infos:
        test_data_path = file_info[1]
        file_name = file_info[0]
        buglist_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/mbart/" + str(
            file_name) + "/buglist_allMR_" + str(file_name)
        gen_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/mbart/" + str(
            file_name) + "/gen_data_allMR_" + str(file_name)
        buglist_no_guidance_data_path = buglist_data_path + "_no_guidance"
        gen_data_no_guidance_path = gen_data_path + "_no_guidance"

        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines = test_data
        bug_list = []
        buglist_no_guidance = []
        single_gen_num = []
        gen_test = []
        gen_test_no_gudiacnce = []
        max_trails = 50
        m_num = 5
        minimum_prob = 1
        while len(lines) > 0:
            sequence = []
            origianl_seed = lines.pop(0).strip()
            print("Original Text:", origianl_seed)
            original_translated_text, original_chosen_beam_tokens_info = mbart_translate_with_token_and_eos_prob(
                mbart_model, mbart_tokenizer, origianl_seed, device=mbart_device)
            original_chosen_beam_tokens_info = original_chosen_beam_tokens_info[
                                               :int(-(
                                                       len(original_chosen_beam_tokens_info) * 0.2))]  # skip the end part of sentence
            # print(original_length)
            diffs = [info[4] for info in original_chosen_beam_tokens_info]
            if len(diffs) == 0:
                pass
            else:
                minimum_prob = min(diffs)
            sequence.append(origianl_seed)
            single_count = 0
            trails = 0


            while len(sequence) > 0 and trails < max_trails:
                english_text = sequence.pop(0)
                words = english_text.split()
                for i in range(m_num):
                    if trails >= max_trails:
                        break
                    trails += 1
                    choice = random.randint(1, 3)
                    rand = random.randint(0, len(words) - 1)
                    if choice == 1:
                        mutated_text = mutate_word_character(english_text, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(english_text, rand, mask_model, mask_tokenizer,
                                                                       mask_device)
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)
                    if mutated_text in gen_test:
                        continue

                    rouge1 = cal_rouge1(origianl_seed, mutated_text)
                    if rouge1 < 0.9:
                        continue
                    gen_test.append(mutated_text)
                    single_count += 1
                    mutated_translated_text, mutated_chosen_beam_tokens_info = mbart_translate_with_token_and_eos_prob(
                        mbart_model,
                        mbart_tokenizer,
                        mutated_text,
                        device=mbart_device)
                    mutated_chosen_beam_tokens_info = mutated_chosen_beam_tokens_info[
                                                      :int(-(len(mutated_chosen_beam_tokens_info) * 0.2))]
                    diffs = [info[4] for info in mutated_chosen_beam_tokens_info]
                    if len(diffs) == 0:
                        pass
                    else:
                        mutated_min_diff = min(diffs)
                    # word align
                    mutated_unaligned_word = word_alignment(mutated_translated_text, mutated_text, args,
                                                            word_align_model,
                                                            word_align_tokenizer, word_align_device)

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
                        if mutated_min_diff < minimum_prob:
                            minimum_prob = mutated_min_diff
                            sequence.append(mutated_text)


        # return 0
        write_gen_num = open(gen_num_path, mode='a', encoding='utf-8')
        write_gen_num_text = str(file_name) + ":" + str(single_gen_num) + "\n"
        write_gen_num.write(write_gen_num_text)
        write_gen_num.close()
        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines2 = test_data
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
                    # choice = 3
                    rand = random.randint(0, len(words) - 1)
                    if choice == 1:
                        mutated_text = mutate_word_character(english_text, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(english_text, rand, mask_model, mask_tokenizer,
                                                                       mask_device)
                    if choice == 3:
                        mutated_text = mutate_punctuations(english_text)
                    if mutated_text in gen_test_no_gudiacnce:
                        continue
                    rouge1 = cal_rouge1(origianl_seed, mutated_text)

                    if rouge1 < 0.9:
                        continue
                    count += 1
                    gen_test_no_gudiacnce.append(mutated_text)
                    mutated_translated_text, mutated_chosen_beam_tokens_info = mbart_translate_with_token_and_eos_prob(
                        mbart_model,
                        mbart_tokenizer,
                        mutated_text,
                        device=mbart_device)

                    mutated_unaligned_word = word_alignment(mutated_translated_text, mutated_text, args,
                                                            word_align_model,
                                                            word_align_tokenizer, word_align_device)
                    print(mutated_unaligned_word)

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


def get_unaligned_non_stop_words(english_sentence, translated_sentence, alignment_sequence):
    english_tokens = english_sentence.split()

    aligned_indices = set(int(pair.split('-')[0]) for pair in alignment_sequence.split())

    word_index_dict = {index: word for index, word in enumerate(english_tokens)
                       if
                       word.lower() not in stop_words and word not in string.punctuation}

    unaligned_words = [word for index, word in word_index_dict.items() if index not in aligned_indices]

    return unaligned_words


def get_files_info(folder_path):
    files_info = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)

            files_info.append((file, full_path))
    return files_info


if __name__ == '__main__':

    args = Args()
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.device = device
    mbart_device = "cuda:1"
    word_align_device = "cuda:2"
    mask_device = "cuda:3"
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

    mbart_model_path = '/home/ubuntu/Desktop/DFuzz4NMT/translator/model/mbart50'
    mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_path).to(torch.device("cuda:1"))
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_path)
    mask_model_path = "/home/ubuntu/Desktop/bert-large-uncased"
    mask_tokenizer = bt.from_pretrained(mask_model_path)
    mask_model = bm.from_pretrained(mask_model_path).to(torch.device("cuda:3"))

    fuzz_omission_mbart(args, mbart_model, mbart_tokenizer, mbart_device, word_align_model,
                           word_align_tokenizer,
                           word_align_device, mask_model, mask_tokenizer, mask_device)
