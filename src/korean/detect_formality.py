from stemmer import stem
from conjugator import *
import tqdm
import os
import numpy as np
import gc

from konlpy.tag import Mecab, Hannanum

punctuations = set(['.', '?', '!', ',', ';', ':'])
gc.enable()

# Takes in a verb, returns 1 if formal (high), or 0 if informal (low). Returns 0.5 if unsure.
def formality(verb):
    if not verb:
        return None
    if verb[-1] in punctuations:
        verb = verb[:-1]
    try:
        high_endings = ['요', '죠']
        low_endings = ['야']
        if verb[-1] in high_endings:
            return 1.
        if verb[-1] in low_endings:
            return 0.
        if not stem(verb): # not a verb
            return 0.5
        forms = conjugation.perform(stem(verb))
        for tense, form in forms[3:]: # first 3 are base forms
            if form[-1] == '?': form = form[:-1]
            if form == verb:
                if tense[-3:] == 'low':
                    return 0.
                elif tense[-4:] == 'high':
                    return 1.
                return 0.5
        return 0.5
    except IndexError:
        #print(verb)
        return 0.5

# def load_word_to_vec_file():
#     word_to_vec_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/cc.ko.300.vec"), "r", encoding="utf-8")
#     # loading throws a UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 3486: invalid continuation byte
#     header = word_to_vec_file.readline()
#     print(header)
#     header_list = header.split()
#     vocab_size = int(header_list[0])
#     embedding_dim = int(header_list[1])
#     embedding_dict = {}
#     for line in tqdm.tqdm(word_to_vec_file, total=vocab_size):
#         line_list = line.split()
#         #print(line_list)
#         word = line_list[0]
#         embedding_vector = np.array([float(x) for x in line_list[1:]])
#         embedding_dict[word] = embedding_vector
#     return embedding_dict, embedding_dim, vocab_size

# load_word_to_vec_file()

# def load_ko_corpus():
#     ko_corpus_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/ko_mini_corpus"), "r", encoding="utf-8")
#     header = ko_corpus_file.readline()
#     #print(header)
#     sentence_list = []
#     for line in tqdm.tqdm(ko_corpus_file):
#         #print(line.strip())
#         line_list = line.strip().split(sep='\t')
#         #print(line_list)
#         sentence = line_list[1]
#         #print(sentence)
#         sentence_list.append(sentence)
#     return sentence_list

def eval_formality(ko_sentences):
    f = open('formality_output.txt', 'w')
    for sentence in ko_sentences:
        word_list = sentence.split()
        #print(word_list[-1])
        formality_level = formality(word_list[-1])
        word_list.append(formality_level)
        f.write('%s, %f' %(sentence, formality_level))

# returns the final verb in a given sentence
# returns None if not found
def detect_final_verb(sentence):
    #verb_list = set(['VV', 'VX', 'XSV', 'VCN', 'VCP', ]
    word_list = sentence.split()
    for i in range(len(word_list)-1, -1, -1):
        tokens = hannanum.pos(word_list[i])
        for token in tokens:
            if token[1] == 'E':
                return word_list[i]
    return None

#ko_sentences = load_ko_corpus()
#eval_formality(ko_sentences)



#f = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/ko_mini_corpus"), "r", encoding="utf-8")
f = open('./corpus.txt', 'r')
sentence_list = f.readlines()
sentence_list = [pair.strip().split('||') for pair in sentence_list]

mecab = Mecab()
#print(sentence_list[4][0])
#print(mecab.pos(sentence_list[4][0]))
#print(mecab.pos('아니요'))

hannanum = Hannanum()

# print(merge('하', 'ㅂ니다'))
# print(formality('니다'))

f = open('./corpus.txt', 'r')
sentence_list = f.readlines()
sentence_list = [pair.strip().split('||') for pair in sentence_list]

formality_list = {0.0: 0, 0.5: 0, 1.0: 0}

#print(detect_final_verb('연구가들이 이미 커피 대체품으로서 음식 대용 과자나 껌에 카페인을 첨가하는 방법을 연구하고 있다고 Archibald는 말했다.'))

f0 = open('f0.txt', 'a')
f05 = open('f05.txt', 'a')
f1 = open('f1.txt', 'a')

for i in range(90000, len(sentence_list)):
#for i in range(len(sentence_list)):
    final_verb = detect_final_verb(sentence_list[i][0])
    if not final_verb:
        continue
    form = formality(final_verb)
    if form == 0.0:
        f0.write(sentence_list[i][0] + '||' + sentence_list[i][1] + '||0||' + final_verb + '\n')
    elif form == 0.5:
        f05.write(sentence_list[i][0] + '||' + sentence_list[i][1] + '||0.5||' + final_verb + '\n')
    elif form == 1.0:
        f1.write(sentence_list[i][0] + '||' + sentence_list[i][1] + '||1||' + final_verb + '\n')
    formality_list[form] += 1
    if i % 1000 == 0:
        print(i)

print(formality_list)

#print(hannanum.pos('의약 연구소는 정부에 과학 문제에 관해 자문하기 위해 의회가 설립 인가를 내어 준 민간 단체인 국립 과학 학회의 부속 단체이다.'))