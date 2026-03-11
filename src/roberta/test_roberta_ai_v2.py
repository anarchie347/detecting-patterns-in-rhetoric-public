from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.roberta.roberta_model_v2 import RobertaModelV2

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_ai_model_v2")
EPS = 1e-6


def report_metrics(y_true: np.ndarray, p_pos: np.ndarray, title: str) -> None:
    y_true = y_true.astype(int)
    p_pos = np.clip(p_pos, EPS, 1 - EPS)
    y_pred = (p_pos >= 0.5).astype(int)

    print(title)
    print("  log_loss:", log_loss(y_true, p_pos))
    print("  roc_auc :", roc_auc_score(y_true, p_pos))
    print("  acc     :", accuracy_score(y_true, y_pred))
    print("  f1      :", f1_score(y_true, y_pred))


def main() -> None:
    df = pd.read_csv(MERGED_DIR / "ai_testing_all.csv", usecols=["text", "label0"])
    texts = df["text"].astype(str).to_numpy()
    y_ai = df["label0"].astype(int).to_numpy()

    model = RobertaModelV2(target_label="label0")
    model.load_model(MODEL_BASE)
    p_ai = model.predict(texts)

    report_metrics(y_ai, p_ai, "AI RoBERTa V2 (label0) metrics")


if __name__ == "__main__":
    main()
