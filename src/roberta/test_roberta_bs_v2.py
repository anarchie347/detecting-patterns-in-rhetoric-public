from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.roberta.roberta_model_v2 import RobertaModelV2

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_bs_model_v2")
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
    df = pd.read_csv(MERGED_DIR / "bullshit_testing_all.csv", usecols=["text", "label1"])
    texts = df["text"].astype(str).to_numpy()
    y_bs = df["label1"].astype(int).to_numpy()

    model = RobertaModelV2(target_label="label1")
    model.load_model(MODEL_BASE)
    p_bs = model.predict(texts)

    report_metrics(y_bs, p_bs, "BS RoBERTa V2 (label1) metrics")


if __name__ == "__main__":
    main()


"""
BS RoBERTa V2 (label1) metrics
  log_loss: 0.027737549288586506
  roc_auc : 0.9995834319526627
  acc     : 0.9938461538461538
  f1      : 0.9938271604938271
"""