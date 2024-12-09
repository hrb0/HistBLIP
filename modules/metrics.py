from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
import spacy
import scispacy
from spacy.tokens import Doc
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np


def key_score_get_keywords(text: str, min_length: int = 3) -> List[str]:
    nlp = spacy.load('en_core_sci_lg')
    doc = nlp(text)
    keywords = []
    for ent in doc.ents:
        if len(ent.text) >= min_length:  
            keywords.append(ent.text.lower())
    return list(set(keywords))  
    
@staticmethod
def key_score_get_jaccard(list1: List[str], list2: List[str]) -> float:
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0
    
def key_score_get_score(ref_text: str, hyp_text: str, min_length: int = 3) -> Tuple[float, List[str], List[str]]:
    ref_keywords = key_score_get_keywords(ref_text, min_length)
    hyp_keywords = key_score_get_keywords(hyp_text, min_length)
        
    score = key_score_get_jaccard(ref_keywords, hyp_keywords)

    return score


def compute_scores_test(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    key_score_list = []
    imgIds = gts.keys()
    for id in imgIds:
        hypo = res[id]
        ref = gts[id]
        key_score = key_score_get_score(ref[0],hypo[0])
        key_score_list.append(key_score)

    eval_res['KEY_SCORE'] = np.mean(key_score_list)
    
    return eval_res

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    
    return eval_res

def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    avg_aucroc = 0
    for i, label in enumerate(label_set):
        res_mlc['AUCROC_' + label] = roc_auc_score(gt[:, i], pred[:, i])
        avg_aucroc += res_mlc['AUCROC_' + label]
    res_mlc['AVG_AUCROC'] = avg_aucroc / len(label_set)

    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc


class MetricWrapper(object):
    def __init__(self, label_set):
        self.label_set = label_set

    def __call__(self, gts, res, gts_mlc, res_mlc):
        eval_res = compute_scores(gts, res)
        eval_res_mlc = compute_mlc(gts_mlc, res_mlc, self.label_set)

        eval_res.update(**eval_res_mlc)
        return eval_res
