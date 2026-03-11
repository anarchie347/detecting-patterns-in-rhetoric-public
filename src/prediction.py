import numpy as np
import os
from pathlib import Path

from scipy.sparse import hstack

from src.roberta.roberta_model import RobertaModel
from src.xgboost.xgb_model import XGBModel

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINED_MODELS_DIR = Path(
    os.environ.get("TRAINED_MODELS_DIR", str(REPO_ROOT / "trained_models"))
).resolve()

XGBOOST_AI_DIR = TRAINED_MODELS_DIR / "xgb_ai_model"
XGBOOST_BS_DIR = TRAINED_MODELS_DIR / "xgb_bs_model"
ROBERTA_AI_DIR = TRAINED_MODELS_DIR / "roberta_ai_model_v2"
ROBERTA_BS_DIR = TRAINED_MODELS_DIR / "roberta_bs_model_v2"

XGB0_NAME = str(XGBOOST_AI_DIR / "xgb_ai_model")
XGB1_NAME = str(XGBOOST_BS_DIR / "xgb_bs_model")

WEIGHT_ROBERTA = 0.6
WEIGHT_XGBOOST = 0.4


def __load_default() -> tuple[XGBModel, XGBModel, RobertaModel, RobertaModel]:
    missing_paths = [
        p
        for p in [
            XGBOOST_AI_DIR,
            XGBOOST_BS_DIR,
            ROBERTA_AI_DIR,
            ROBERTA_BS_DIR,
        ]
        if not p.exists()
    ]
    if missing_paths:
        missing = "\n".join(f"- {p}" for p in missing_paths)
        raise FileNotFoundError(
            "Missing trained model directories. Expected under:\n"
            f"{TRAINED_MODELS_DIR}\n"
            "Missing:\n"
            f"{missing}"
        )

    xgb0 = XGBModel()
    xgb1 = XGBModel()
    rob0 = RobertaModel(target_label="label0")
    rob1 = RobertaModel(target_label="label1")

    xgb0.load_model(XGB0_NAME)
    xgb1.load_model(XGB1_NAME)
    rob0.load_model("unused", model_dir=str(ROBERTA_AI_DIR))
    rob1.load_model("unused", model_dir=str(ROBERTA_BS_DIR))

    return xgb0, xgb1, rob0, rob1


def _xgb_predict_label1(model: XGBModel, texts: np.ndarray) -> np.ndarray:
    x_tfidf = model.vectorizer.transform(texts)
    total_weight = np.array(x_tfidf.sum(axis=1)) + 1e-9
    x_mean = (x_tfidf.dot(model.means).reshape(-1, 1)) / total_weight
    x_sd = (x_tfidf.dot(model.sds).reshape(-1, 1)) / total_weight
    X = hstack([x_tfidf, x_mean, x_sd])
    return model.classifer.predict_proba(X)[:, 1].astype(np.float64)


def _predict_weighted(texts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xgb0, xgb1, rob0, rob1 = __load_default()

    p_xgb0 = _xgb_predict_label1(xgb0, texts)
    p_xgb1 = _xgb_predict_label1(xgb1, texts)
    p_rob0 = rob0.predict(texts)
    p_rob1 = rob1.predict(texts)

    p_ai = WEIGHT_ROBERTA * p_rob0 + WEIGHT_XGBOOST * p_xgb0
    p_bs = WEIGHT_ROBERTA * p_rob1 + WEIGHT_XGBOOST * p_xgb1
    return p_ai, p_bs

def scoreWhole(text : str) -> tuple[int, int]:
    [result_ai], [result_bs] = _predict_weighted(np.array([text]))
    return int(result_bs * 100), int(result_ai * 100)

def _sentence_idxs(text: str) -> list[tuple[int,int]]:
    """
    returns a list of tuples containing start (inclusive) and end (exclusive) indexes for the sentences in a text
    
    :param text: text to index
    :type text: str
    :return: list of tuples
    :rtype: list[tuple[int, int]]
    """
    delims = ['.', '!', '?', '\n']

    delim_idxs = [i for i,c in enumerate(text) if c in delims]
    delim_idxs.append(len(text))
    delim_idxs.insert(0,-1)
    sentence_boundaries =  [ (delim_idxs[i-1]+1, d_idx) for i,d_idx in enumerate(delim_idxs) if i > 0]

    return sentence_boundaries

def _text_skip_sentence(text : str, skipIdx : tuple[int,int]) -> str:
    """
    Based on a sentence start and end index tuple, return the text without that sentence
    
    :param text: the text
    :type text: str
    :param skipIdx: the start (inclusive) and end (exclusive) indexes
    :type skipIdx: tuple[int, int]
    :return: The text without the sentence
    :rtype: str
    """
    return text[:skipIdx[0]] + text[skipIdx[1]+1:]



def scoreSentences(text: str, overall_bs : int, overall_ai : int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    BS_THRESHOLD = 1
    AI_THRESHOLD = 1
    
    sentence_boundaries = _sentence_idxs(text)
    skip_one = np.array([_text_skip_sentence(text,idxs) for idxs in sentence_boundaries])

    results_ai, results_bs = _predict_weighted(skip_one)

    results_bs = overall_bs - (results_bs * 100)
    results_ai = overall_ai - (results_ai * 100)

    bs_highlight = []
    ai_highlight = []

    for i, boundaries in enumerate(sentence_boundaries):
        if results_bs[i] > BS_THRESHOLD:
            bs_highlight.append(boundaries)
        if results_ai[i] > AI_THRESHOLD:
            ai_highlight.append(boundaries) 

    """
    bs_list : list[int] = np.round(results_bs).astype(int).tolist()
    ai_list : list[int] = np.round(results_ai).astype(int).tolist()
    return list(zip(bs_list, sentence_boundaries)), list(zip(ai_list, sentence_boundaries))
    """
    return bs_highlight, ai_highlight
