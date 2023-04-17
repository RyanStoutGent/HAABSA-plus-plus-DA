import string
from transformers import BertTokenizer
from transformers import pipeline
import torch
import re
import random as rd

BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
unmasker = pipeline(task='fill-mask', model='bert-base-uncased',topk = 2)

def file_maker(in_file, out_file):
    print('Starting BERT-augmentation')
    with open(in_file, 'r') as in_f, open(out_file, 'w+', encoding='utf-8') as out_f:
        lines = in_f.readlines()
        for i in range(0, len(lines) - 1, 3):
            old_sentence = lines[i].strip()
            target = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()
            new_sentence = augment_sentence(old_sentence,target)
            out_f.writelines([old_sentence + '\n', target + '\n', sentiment + '\n'])
            out_f.writelines([new_sentence + '\n', target + '\n', sentiment + '\n'])
    return out_file

def augment_sentence(in_sentence, in_target):

    words = tokenizer.tokenize(in_sentence)
    tar = re.findall(r'\w+|[^\s\w]+', in_target)
    for word in tar:
        word = tokenizer.tokenize(word)
    tar_length = len(tar)

    masked_sen = []
    ind = 0

    for wrd in words:
        j = words.index(wrd)
        if wrd == '$' and words[j+1]=='t' and words[j+2]=='$':
            ind = words.index(wrd)

    masked_sen.extend(words[:ind])
    masked_sen.extend(tar)
    masked_sen.extend(words[(ind+3):])
    augmentend_sentence = []

    j=0
    number_not_words = 0
    while j < len(words):
        if words[j] == '$' and words[j+1]=='t' and words[j+2]=='$':
            j+=3
            number_not_words +=3
        elif words[j] in string.punctuation:
            j += 1
            number_not_words +=1
        else:
            j += 1

    mask_prob = 0.2
    total_masks = min(1,int(round((len(words)-number_not_words)*mask_prob)))
    amount_masked = 0
    vocab = tokenizer.vocab
    real_percentage = mask_prob / ( (len(words)-number_not_words)/len(words) )

    i=0
    while i < len(words):
        if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
            augmentend_sentence.append('$T$')
            i+=3
        elif words[i] in string.punctuation:
            augmentend_sentence.append(words[i])
            i += 1
        else:
            prob1 = rd.random()
            if prob1 <= real_percentage:
                prob2 = rd.random()
                if prob2 <= 1:
                    amount_masked += 1
                    cur_sent = masked_sen.copy()
                    masked_word = words[i]
                    if i < ind:
                        cur_sent[i] = '[MASK]'
                    else:
                        cur_sent[i-(3-tar_length)] = '[MASK]'
                    results = unmasker(' '.join(cur_sent))
                    predicted_words = []

                    for result in results:
                        token_id = result['token']
                        token_str = tokenizer.decode([token_id])
                        predicted_words.append(token_str)
                    if predicted_words[1]==masked_word:
                        augmentend_sentence.append(predicted_words[0])
                        i += 1
                    else:
                        augmentend_sentence.append(predicted_words[1])
                        i += 1
                elif 0.8 < prob2 <= 0.9:
                    amount_masked += 1
                    random_token = rd.choice(list(vocab.keys()))
                    augmentend_sentence.append(random_token)
                    i += 1
                else:
                    augmentend_sentence.append(words[i])
                    i += 1
            else:
                augmentend_sentence.append(words[i])
                i+=1

    augmentend_sentence_str = ' '.join(augmentend_sentence)
    print(augmentend_sentence_str)
    return augmentend_sentence_str

'''
    while i < len(words):
        if amount_masked >= total_masks and i == len(words)-1:
            break;
        elif i == len(words)-1 and amount_masked < total_masks:
            i=0
        else:
            if words[i] == '$' and words[i+1]=='t' and words[i+2]=='$':
                augmentend_sentence.append('$T$')
                i+=3
            elif words[i] in string.punctuation:
                augmentend_sentence.append(words[i])
                i += 1
            else:
                if rd.random() <= 0.15:
                    prob = rd.random()
                    if prob <= 0.8:
                        amount_masked += 1
                        cur_sent = masked_sen.copy()
                        masked_word = words[i]
                        if i < ind:
                            cur_sent[i] = '[MASK]'
                        else:
                            cur_sent[i-(3-tar_length)] = '[MASK]'
                        results = unmasker(' '.join(cur_sent))
                        predicted_words = []

                        for result in results:
                            token_id = result['token']
                            token_str = tokenizer.decode([token_id])
                            predicted_words.append(token_str)
                        if predicted_words[1]==masked_word:
                            augmentend_sentence.append(predicted_words[0])
                        else:
                            augmentend_sentence.append(predicted_words[1])
                    elif 0.8 < prob <= 0.9:
                        amount_masked += 1
                        random_token = rd.choice(list(vocab.keys()))
                        augmentend_sentence.append(random_token)
                    else:
                        augmentend_sentence.append(words[i])
                else:
                    augmentend_sentence.append(words[i])
                    i+=1

    augmentend_sentence_str = ' '.join(augmentend_sentence)
    print(augmentend_sentence_str)
    return augmentend_sentence_str
'''
