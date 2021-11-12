import nltk
import os
import numpy as np
import tqdm

def calculate_bleu_score(hypothesis, reference):
    """
    Input:  hypothesis: list(str)
            reference:  list(str)
    Output: BLEU score: float
    """
    min_length = min(len(hypothesis), len(reference))
    if min_length < 4:
        weights = [1.0 / min_length for i in range(0, min_length)]
        score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = weights)
        return score
    else:
        score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
        return score

def calculate_corpus_bleu_score(hypotheses, references):
    """
    Input:  hypotheses: list(list(list(str)))
            references: list(list(str))
    Output: BLEU score: float
    """
    score = nltk.translate.bleu_score.sentence_bleu(references, hypotheses)
    return score

def load_word_to_vec_file():
    word_to_vec_file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/cc.ja.300.vec"), "r", encoding="utf-8")
    header = word_to_vec_file.readline()
    header_list = header.split()
    vocab_size = int(header_list[0])
    embedding_dim = int(header_list[1])
    embedding_dict = {}
    for line in tqdm.tqdm(word_to_vec_file, total=vocab_size):
        line_list = line.split()
        word = line_list[0]
        embedding_vector = np.array([float(x) for x in line_list[1:]])
        embedding_dict[word] = embedding_vector
    return embedding_dict, embedding_dim, vocab_size

def calculate_sentence_similarity(hypothesis, reference, embedding_dict, embedding_dim):
    """
    Input:  hypotheses: list(str)
            references: list(str)
    Output: similarity: float
    """
    hypothesis_embedding = np.zeros(embedding_dim)
    reference_embedding = np.zeros(embedding_dim)
    for hypothesis_word in hypothesis:
        hypothesis_embedding += embedding_dict[hypothesis_word]
    for reference_word in reference:
        reference_embedding += embedding_dict[reference_word]
    hypothesis_embedding = hypothesis_embedding / len(hypothesis)
    reference_embedding = reference_embedding / len(reference)
    similarity = np.dot(hypothesis_embedding, reference_embedding)/(np.linalg.norm(hypothesis_embedding) * np.linalg.nomr(reference_embedding))
    return similarity
