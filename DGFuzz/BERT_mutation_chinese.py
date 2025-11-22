import torch
import random
import jieba
from transformers import BertTokenizer, BertForMaskedLM

import torch
import random
from transformers import BertTokenizer, BertForMaskedLM

def predict_single_char_by_index(sentence, char_index, model, tokenizer, device, top_k=10):

    if char_index < 0 or char_index >= len(sentence):
        return sentence


    masked_sentence = sentence[:char_index] + tokenizer.mask_token + sentence[char_index+1:]
    print("masked sentence: "+str(masked_sentence))


    inputs = tokenizer(masked_sentence, return_tensors='pt').to(torch.device(device))
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits


    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    if len(mask_token_index) == 0:
        print("no mask position")
        return sentence


    token_logits = logits[0, mask_token_index[0]]
    top_token_ids = torch.topk(token_logits, top_k).indices.tolist()
    candidates = [tokenizer.convert_ids_to_tokens([tid])[0] for tid in top_token_ids]
    candidates = [c for c in candidates if all('\u4e00' <= ch <= '\u9fff' for ch in c)]

    if not candidates:
        print("no candidates")
        return sentence


    predicted_char = random.choice(candidates)



    new_sentence = sentence[:char_index] + predicted_char + sentence[char_index+1:]

    return new_sentence



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "/home/ubuntu/Desktop/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path).to(device)

    sentence = "他打算明天去北京出差。"
    char_index = 6  # 替换 “北”

    for _ in range(10):
        new_sentence = predict_single_char_by_index(sentence, char_index, model, tokenizer, device)
        print("新句：", new_sentence)
