import os
import random
import string
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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


def helsinki_translate_with_token_and_eos_prob(model, tokenizer, input_text, device, beam_size=5):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(torch.device(device))

    outputs = model.generate(inputs, max_length=256, num_beams=beam_size, return_dict_in_generate=True,
                             output_scores=True)
    decoded_texts = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens=True)
    beam_scores = outputs["scores"]

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


def fuzz_omission_helsinki(args, helsinki_model, helsinki_tokenizer, helsinki_device, word_align_model,
                           word_align_tokenizer, word_align_device, mask_model, mask_tokenizer, mask_device):
    file_path = "/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/dataset"
    choices = [1,2,3]
    gen_num_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/gen_num_result"

    for choice in choices:
        if choice == 1:
            mutation = "MO1"
        if choice == 2:
            mutation = "MO2"
        if choice == 3:
            mutation = "MO3"

        file_infos = get_files_info(file_path)
        for file_info in file_infos:
            test_data_path = file_info[1]
            file_name = file_info[0]
            print("start")
            print(test_data_path)
            print(file_name)
            buglist_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki/" + str(
                file_name) + "/buglist_" + str(file_name) + "_" + mutation
            gen_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki/" + str(
                file_name) + "/gen_data_" + str(file_name) + "_" + mutation
            with open(gen_num_path, 'r', encoding='utf-8') as file:
                gen_num_data = file.readlines()
            for gen_num_info in gen_num_data:
                name = gen_num_info.split(":")[0]
                print(name)
                print(file_name)
                if name == file_name:
                    gen_num = json.loads(gen_num_info.split(":")[1])
            with open(test_data_path, 'r', encoding='utf-8') as file:
                test_data = file.readlines()
                lines = test_data
            bug_list = []
            gen_test = []
            while len(lines) > 0:
                origianl_seed = lines.pop(0).strip()
                num = gen_num.pop(0)
                count = 0
                words = origianl_seed.split()
                while count < num:
                    rand = random.randint(0, len(words) - 1)
                    if choice == 1:
                        mutated_text = mutate_word_character(origianl_seed, rand)
                    if choice == 2:
                        mutated_text = predict_masked_word_by_position(origianl_seed, rand, mask_model,
                                                                       mask_tokenizer,
                                                                       mask_device)
                    if choice == 3:
                        mutated_text = mutate_punctuations(origianl_seed)
                    if mutated_text in gen_test:
                        continue

                    rouge1 = cal_rouge1(origianl_seed, mutated_text)
                    print("rouge1:  " + str(rouge1))
                    if rouge1 < 0.9:
                        continue
                    gen_test.append(mutated_text)
                    count += 1
                    mutated_translated_text, mutated_chosen_beam_tokens_info = helsinki_translate_with_token_and_eos_prob(
                        helsinki_model,
                        helsinki_tokenizer,
                        mutated_text,
                        device=helsinki_device)
                    # word align
                    mutated_unaligned_word = word_alignment(mutated_translated_text, mutated_text, args,
                                                            word_align_model,
                                                            word_align_tokenizer, word_align_device)
                    print(mutated_unaligned_word)

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



            write_guidance_all = open(gen_data_path, mode='a', encoding='utf-8')
            for item in gen_test:
                if '\n' not in item:
                    item = item + '\n'
                write_guidance_all.write(item)
            write_guidance_all.close()



def get_unaligned_non_stop_words(english_sentence, translated_sentence, alignment_sequence):
    english_tokens = english_sentence.split()
    translated_tokens = translated_sentence.split()

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
    helsinki_device = "cuda:1"
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

    helsinki_model_path = '/home/ubuntu/Desktop/DFuzz4NMT/translator/model/Helsinki'
    helsinki_model = MarianMTModel.from_pretrained(helsinki_model_path).to(torch.device("cuda:1"))
    helsinki_tokenizer = MarianTokenizer.from_pretrained(helsinki_model_path)
    mask_model_path = "/home/ubuntu/Desktop/bert-large-uncased"
    mask_tokenizer = bt.from_pretrained(mask_model_path)
    mask_model = bm.from_pretrained(mask_model_path).to(torch.device("cuda:3"))

    fuzz_omission_helsinki(args, helsinki_model, helsinki_tokenizer, helsinki_device, word_align_model,
                           word_align_tokenizer,
                           word_align_device, mask_model, mask_tokenizer, mask_device)
