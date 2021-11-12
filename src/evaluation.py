import nltk

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
