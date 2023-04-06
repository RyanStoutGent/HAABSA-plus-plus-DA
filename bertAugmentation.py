import string

from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
import torch
import re
import random as rd


BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
unmasker = pipeline(task='fill-mask', model='bert-base-uncased',topk = 2)

#replacement_per = FLAGS.

def main(in_file, out_file):
    print('Starting BERT-augmentation')
    with open(in_file, 'r') as in_f, open(out_file, 'w+', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in range(0, len(lines) - 1, 3):
            old_sentences = lines[i]
            targets = lines[i + 1] #strip?
            sentiments = lines[i + 2]
def augment_sentence(in_sentence, in_target):
    words = tokenizer.tokenize(in_sentence)
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    for word in tar:
        word = tokenizer.tokenize(word)
    tar_length = len(tar)
    augmentend_sentence = []
    i=0
    while i < len(words):
        #if word=='[CLS]' or word=='[SEP]' or word=='[PAD]'
        if words[i:i+tar_length] == tar:
            augmentend_sentence.append('$T$')
            i += tar_length
        elif words[i] in string.punctuation:
            augmentend_sentence.append(words[i])
            i += 1
        else:
            prob = rd.random()
            if prob <= 0.8:
                cur_sent = words.copy()
                masked_word = words[i]
                cur_sent[i] = '[MASK]'
                cur_sent_input = ' '.join(cur_sent)
                results = unmasker(cur_sent_input)
                predicted_words = []
                print(masked_word)
                for result in results:
                    token_id = result['token']
                    token_str = tokenizer.decode([token_id])
                    predicted_words.append(token_str)
                if predicted_words[1]==masked_word:
                    augmentend_sentence.append(predicted_words[0])
                    print(predicted_words[0])
                else:
                    augmentend_sentence.append(predicted_words[1])
                    print(predicted_words[1])
            else:
                augmentend_sentence.append(words[i])
            i+=1
    print(in_sentence)
    print(augmentend_sentence)
    return augmentend_sentence





    '''
    for i in range(0, len(lines)-1, 3):
            sentences = lines[i]
            targets = lines[i+1]
            sentiments = lines[i+2]
    '''