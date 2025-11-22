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
# nltk.download('stopwords')
from nltk.corpus import stopwords
from zhon.hanzi import punctuation as ch_punctuation

# punctuation_set = ["!","”","#","$","%","&","(", ")" ,"∗" ,"+",",", "−",".","/",":",";", "<","=",">","?","@","[", "]", "ˆ","_","‘","{", "}"]
# nltk.download('stopwords')
stop_words =[]
with open('/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/hit_stopwords.txt', 'r', encoding='utf-8') as f:
    for line in f:
        stop_words.append(line.strip())

from Fuzz4SeMiss.calculate_rough1 import  rouge1_ch
from word_align.awesome_align import modeling
from word_align.awesome_align.configuration_bert import BertConfig
from word_align.awesome_align.modeling import BertForMaskedLM
from word_align.awesome_align.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer
# from Fuzz4SeMiss.eos_mutation import mutate_word_character, mutate_punctuations
from eos_mutation_ch import mutate_chinese_punctuations
from bert_mutation_ch import predict_single_char_by_index
from Fuzz4SeMiss.bert_mutation import predict_masked_word_by_position
from word_align.data_tokenize import transform
from word_align.word_align import set_seed


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


def helsinki_translate_with_token_prob_ch2en(model, tokenizer, input_text, device, beam_size=5):
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(torch.device(device))

    outputs = model.generate(inputs, max_length=512, num_beams=beam_size, return_dict_in_generate=True,
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


# Instantiate the model and tokenizer

# Process input text
def word_alignment(chinese_text, english_text, args, word_align_model,
                   word_align_tokenizer, word_align_device):
    awesome_ch_token, awesome_en_token = transform(chinese_text, english_text)
    align_text = awesome_en_token + " ||| " + awesome_ch_token
    input = open("/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/input", mode='w', encoding='utf-8')
    input.write(align_text)
    input.close()
    args.output_file = "/home/ubuntu/Desktop/DFuzz4NMT/word_align/test/detect_omission.txt"
    word_align.word_align(args, word_align_model, word_align_tokenizer, word_align_device)
    output_alignment = open(args.output_file, mode='r', encoding='utf-8').readline()

    chinese_unaligned_word = get_unaligned_non_stop_words_chinese(english_text,chinese_text,output_alignment,stop_words)
    return chinese_unaligned_word


def fuzz_omission_helsinki_en2ch_ch2en(args, helsinki_en2ch_model, helsinki_en2ch_tokenizer, helsinki_en2ch_device, word_align_model,
                          word_align_tokenizer, word_align_device, mask_model, mask_tokenizer, mask_device):
    all_8_bug_num = 0
    all_8_bug_num_no_guidance = 0
    all_8_gen_num = 0
    all_8_gen_num_no_guidance = 0
    all_8_cov = 0
    all_8_cov_no_guidance = 0
    all_8_record_results = []
    all_8_record_results_no_guidance = []
    result_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki_results_ch2en_"
    gen_num_path= "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki_gen_num_result_ch2en_"
    file_path = "/home/ubuntu/Desktop/DFuzz4NMT/Fuzz4SeMiss/omission_dataset_chinese"
    file_infos = get_files_info(file_path)
    for file_info in file_infos:
        test_data_path = file_info[1]
        file_name = file_info[0]
        print("start")

        buglist_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki_ch2en/" + str(
            file_name) + "/buglist_allMR_t=4_ch2en_" + str(file_name)
        gen_data_path = "/home/ubuntu/Desktop/DFuzz4NMT/experiment/helsinki_ch2en/" + str(
            file_name) + "/gen_data_allMR_t=4_ch2en_" + str(file_name)
        buglist_no_guidance_data_path = buglist_data_path + "_no_guidance"
        gen_data_no_guidance_path = gen_data_path + "_no_guidance"

        with open(test_data_path, 'r', encoding='utf-8') as file:
            test_data = file.readlines()
            lines = test_data
        bug_list = []
        buglist_no_guidance = []
        single_gen_num = []
        bug_num = 0
        bug_num_no_guidance = 0
        total_count = 0
        total_count_no_guidance = 0
        gen_test = []
        gen_test_no_gudiacnce = []
        max_trails = 50
        cov = 0
        cov_no_guidance = 0
        record_results = []
        record_results_no_guidance = []
        m_num = 5
        minimum_prob = 1
        while len(lines) > 0:
            sequence = []
            origianl_seed = lines.pop(0).strip()

            original_translated_text, original_chosen_beam_tokens_info = helsinki_translate_with_token_prob_ch2en(
                helsinki_en2ch_model, helsinki_en2ch_tokenizer, origianl_seed, device=helsinki_en2ch_device)

            original_chosen_beam_tokens_info = original_chosen_beam_tokens_info[
                                               :int(-(
                                                           len(original_chosen_beam_tokens_info) * 0.2))]  # skip the end part of sentence

            diffs = [info[4] for info in original_chosen_beam_tokens_info]

            if len(diffs) == 0:
                pass
            else:
                minimum_prob = min(diffs)

            sequence.append(origianl_seed)
            single_count = 0
            trails = 0
            flag = False

            while len(sequence) > 0 and trails < max_trails:
                chinese_text = sequence.pop(0)
                words = list(chinese_text)

                for i in range(m_num):
                    if trails >= max_trails:
                        break
                    trails += 1
                    choice = random.randint(2, 3)

                    rand = random.randint(0, len(words) - 1)

                    if choice == 2:
                        mutated_text = predict_single_char_by_index(chinese_text,rand,mask_model,mask_tokenizer,mask_device,10)

                    if choice == 3:
                        mutated_text = mutate_chinese_punctuations(chinese_text)

                    if mutated_text in gen_test:
                        continue



                    rouge1 = rouge1_ch(origianl_seed, mutated_text)

                    if rouge1 < 0.9:
                        continue
                    gen_test.append(mutated_text)
                    single_count += 1
                    total_count += 1
                    all_8_gen_num += 1
                    mutated_translated_text, mutated_chosen_beam_tokens_info = helsinki_translate_with_token_prob_ch2en(
                        helsinki_en2ch_model,
                        helsinki_en2ch_tokenizer,
                        mutated_text,
                        device=helsinki_en2ch_device)
                    mutated_chosen_beam_tokens_info = mutated_chosen_beam_tokens_info[
                                                      :int(-(len(mutated_chosen_beam_tokens_info) * 0.2))]
                    diffs = [info[4] for info in mutated_chosen_beam_tokens_info]
                    if len(diffs) == 0:
                        pass
                    else:
                        mutated_min_diff = min(diffs)

                    # word align
                    mutated_unaligned_word = word_alignment(mutated_text,mutated_translated_text,  args,
                                                            word_align_model,
                                                            word_align_tokenizer, word_align_device)


                    if len(mutated_unaligned_word) > 4:
                        flag = True
                        bug_num += 1
                        all_8_bug_num += 1

                        bug_list.append(mutated_text)
                        bug_list_data = open(buglist_data_path, mode='a', encoding='utf-8')
                        if '\n' not in mutated_text:
                            mutated_text = mutated_text + '\n'
                        if '\n' not in mutated_translated_text:
                            mutated_translated_text = mutated_translated_text + '\n'
                        bug_list_data.write(mutated_text + str(mutated_unaligned_word)+ mutated_translated_text)
                        bug_list_data.close()
                    else:
                        if mutated_min_diff < minimum_prob:
                            minimum_prob = mutated_min_diff
                            sequence.append(mutated_text)

                    if total_count % 10 == 0:
                        record_results.append([total_count, bug_num])
                    if all_8_gen_num % 10 == 0:
                        all_8_record_results.append([all_8_gen_num, all_8_bug_num])
            if flag:
                cov += 1
                all_8_cov += 1
            single_gen_num.append(single_count)

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
            flag_no_guidance = False
            while count < num:
                if len(sequence) == 0:
                    sequence.append(origianl_seed)
                chinese_text = sequence.pop(0).strip()
                words = list(chinese_text)
                for i in range(m_num):
                    if count >= num:
                        break
                    choice = random.randint(2, 3)
                    # choice = 3
                    rand = random.randint(0, len(words) - 1)
                    if choice == 2:
                        mutated_text = predict_single_char_by_index(chinese_text, rand, mask_model, mask_tokenizer,
                                                                    mask_device, 10)
                    if choice == 3:
                        mutated_text = mutate_chinese_punctuations(chinese_text)
                    if mutated_text in gen_test_no_gudiacnce:
                        continue
                    rouge1 = rouge1_ch(origianl_seed, mutated_text)

                    if rouge1 < 0.9:
                        continue
                    count += 1
                    total_count_no_guidance += 1
                    all_8_gen_num_no_guidance += 1
                    gen_test_no_gudiacnce.append(mutated_text)
                    mutated_translated_text, mutated_chosen_beam_tokens_info = helsinki_translate_with_token_prob_ch2en(
                        helsinki_en2ch_model,
                        helsinki_en2ch_tokenizer,
                        mutated_text,
                        device=helsinki_en2ch_device)

                    mutated_unaligned_word_noguidance = word_alignment( mutated_text, mutated_translated_text, args,
                                                            word_align_model,
                                                            word_align_tokenizer, word_align_device)


                    if len(mutated_unaligned_word_noguidance) > 4:
                        bug_num_no_guidance += 1
                        all_8_bug_num_no_guidance += 1
                        flag_no_guidance = True

                        buglist_no_guidance.append(mutated_text)
                        buglist_no_guidance_data = open(buglist_no_guidance_data_path, mode='a', encoding='utf-8')
                        if '\n' not in mutated_text:
                            mutated_text = mutated_text + '\n'
                        if '\n' not in mutated_translated_text:
                            mutated_translated_text = mutated_translated_text + '\n'
                        buglist_no_guidance_data.write(mutated_text +str(mutated_unaligned_word_noguidance)+ mutated_translated_text)
                        buglist_no_guidance_data.close()
                    else:
                        sequence.append(mutated_text)
                    if total_count_no_guidance % 10 == 0:
                        record_results_no_guidance.append([total_count_no_guidance, bug_num_no_guidance])
                    if all_8_gen_num_no_guidance % 10 == 0:
                        all_8_record_results_no_guidance.append([all_8_gen_num_no_guidance, all_8_bug_num_no_guidance])
            if flag_no_guidance:
                cov_no_guidance += 1
                all_8_cov_no_guidance += 1


        record_results.append([total_count, bug_num])
        record_results_no_guidance.append([total_count_no_guidance, bug_num_no_guidance])

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
                    str(buglist_data_path) + "\n" + "bug_num:  " + str(bug_num) + "\tgen_num:  " + str(len(gen_test))
                    + "\tcov:  " + str(cov) + "\n" + str(record_results) + "\n\n"
                    + "bug_num_no_guidance:  " + str(bug_num_no_guidance) + "\tgen_num_no_guidance:  " + str(
                len(gen_test_no_gudiacnce))
                    + "\tcov_no_guidance:  " + str(cov_no_guidance) + "\n" + str(
                record_results_no_guidance) + "\n" + str(write_gen_num) + "\n\n\n\n"
                    )
        write_results.write(write_results_text)
        write_results.close()

    all_8_record_results.append([all_8_gen_num, all_8_bug_num])
    all_8_record_results_no_guidance.append([all_8_gen_num_no_guidance, all_8_bug_num_no_guidance])
    all_8_write_results = open(result_path, mode='a', encoding='utf-8')
    all_8_write_results_text = (
            "all_8:" + "\n" + "all_8_bug_num:  " + str(all_8_bug_num) + "\tall_8_gen_num:  " + str(all_8_gen_num)
            + "\tall_8_cov:  " + str(all_8_cov) + "\n" + str(all_8_record_results) + "\n\n"
            + "all8_no_guidance_bug_num:  " + str(all_8_bug_num_no_guidance) + "\tgen_num_no_guidance:  " + str(
        all_8_gen_num_no_guidance)
            + "\tcov_no_guidance:  " + str(all_8_cov_no_guidance) + "\n" + str(
        all_8_record_results_no_guidance) + "\n\n\n\n"
    )
    all_8_write_results.write(all_8_write_results_text)
    all_8_write_results.close()


def get_unaligned_non_stop_words_chinese(english_sentence, chinese_sentence, alignment_sequence, chinese_stopwords):
    # english_tokens = english_sentence.split()
    ch_tokens,en_tokens =transform(chinese_sentence,english_sentence)
    ch_tokens=ch_tokens.strip().split()
    indexed_tokens = {i: ch_token for i, ch_token in enumerate(ch_tokens)}


    aligned_indices = set(int(pair.split('-')[1]) for pair in alignment_sequence.split())

    un_aligned_indices=[]
    un_aligned_ch_tokens=[]
    for i in range(len(ch_tokens)):
        if i not in aligned_indices:
            un_aligned_indices.append(i)


    for item in un_aligned_indices:

        if ch_tokens[item] not in chinese_stopwords and ch_tokens[item]not in ch_punctuation:
            un_aligned_ch_tokens.append(ch_tokens[item])

    return un_aligned_ch_tokens


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
    helsinki_en2ch_device = "cuda:1"
    word_align_device = "cuda:2"
    mask_device = "cuda:3"
    # Set seed
    set_seed(args)
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

    helsinki_en2ch_model_path = "/home/ubuntu/Desktop/DFuzz4NMT/translator/model/helsinki-ch2en"
    helsinki_en2ch_model = AutoModelForSeq2SeqLM.from_pretrained(helsinki_en2ch_model_path).to(torch.device("cuda:1"))
    helsinki_en2ch_tokenizer = AutoTokenizer.from_pretrained(helsinki_en2ch_model_path)
    mask_model_path = "/home/ubuntu/Desktop/bert-base-chinese"
    mask_tokenizer = bt.from_pretrained(mask_model_path)
    mask_model = bm.from_pretrained(mask_model_path).to(torch.device("cuda:3"))

    fuzz_omission_helsinki_en2ch_ch2en(args, helsinki_en2ch_model, helsinki_en2ch_tokenizer, helsinki_en2ch_device, word_align_model,
                          word_align_tokenizer,
                          word_align_device, mask_model, mask_tokenizer, mask_device)
