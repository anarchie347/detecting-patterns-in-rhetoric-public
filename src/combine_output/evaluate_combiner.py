from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.combine_output.combiner_both import load_combiner

EPS = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
COMBINER_MODEL_DIR = Path(__file__).resolve().parent / "combiner_model"
XGBOOST_DIR = REPO_ROOT / "src" / "xgboost"
XGBOOST_AI_DIR = XGBOOST_DIR / "xgb_ai_model"
XGBOOST_BS_DIR = XGBOOST_DIR / "xgb_bs_model"

ROBERTA_AI_DIR = REPO_ROOT / "src" / "roberta" / "roberta_ai_model"
ROBERTA_BS_DIR = REPO_ROOT / "src" / "roberta" / "roberta_bs_model"

XGB0_NAME = str(XGBOOST_AI_DIR / "xgb_ai_model")
XGB1_NAME = str(XGBOOST_BS_DIR / "xgb_bs_model")
META_PATH = str(COMBINER_MODEL_DIR / "combiner.joblib")


def report_metrics(y_true: np.ndarray, p_pos: np.ndarray, title: str) -> None:
    y_true = y_true.astype(int)
    p_pos = np.clip(p_pos, EPS, 1 - EPS)
    y_pred = (p_pos >= 0.5).astype(int)

    print(f"\n{title}")
    print(f"  log_loss : {log_loss(y_true, p_pos):.4f}")
    print(f"  roc_auc  : {roc_auc_score(y_true, p_pos):.4f}")
    print(f"  accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  f1       : {f1_score(y_true, y_pred):.4f}")


def main() -> None:
    df_ai = pd.read_csv(MERGED_DIR / "ai_testing_all.csv")
    texts_ai = df_ai["text"].astype(str).to_numpy()
    y0 = df_ai["label0"].astype(int).to_numpy()

    df_bs = pd.read_csv(MERGED_DIR / "bullshit_testing_all.csv")
    texts_bs = df_bs["text"].astype(str).to_numpy()
    y1 = df_bs["label1"].astype(int).to_numpy()

    print(f"AI test set size: {len(texts_ai)}")
    print(f"BS test set size: {len(texts_bs)}")

    combiner = load_combiner(
        xgb0_name=XGB0_NAME,
        xgb1_name=XGB1_NAME,
        meta_path=META_PATH,
        rob0_dir=str(ROBERTA_AI_DIR),
        rob1_dir=str(ROBERTA_BS_DIR),
    )

    p0_comb, p1_comb = combiner.predict_separate(texts_ai, texts_bs)

    # Base-model references
    p0_xgb = combiner._xgb_predict(combiner.xgb0, texts_ai)
    p1_xgb = combiner._xgb_predict(combiner.xgb1, texts_bs)
    p0_rob = combiner.rob0.predict(texts_ai)
    p1_rob = combiner.rob1.predict(texts_bs)

    report_metrics(y0, p0_xgb, "AI Task - XGBoost only")
    report_metrics(y0, p0_rob, "AI Task - RoBERTa only")
    report_metrics(y0, p0_comb, "AI Task - Combiner")

    report_metrics(y1, p1_xgb, "BS Task - XGBoost only")
    report_metrics(y1, p1_rob, "BS Task - RoBERTa only")
    report_metrics(y1, p1_comb, "BS Task - Combiner")


if __name__ == "__main__":
    main()


"""
AI Task - XGBoost only
  log_loss : 0.2484
  roc_auc  : 0.9770
  accuracy : 0.8943
  f1       : 0.8841

AI Task - RoBERTa only
  log_loss : 0.1326
  roc_auc  : 0.9980
  accuracy : 0.9875
  f1       : 0.9875

AI Task - Combiner
  log_loss : 0.0933
  roc_auc  : 0.9989
  accuracy : 0.9875
  f1       : 0.9875

BS Task - XGBoost only
  log_loss : 0.1094
  roc_auc  : 0.9945
  accuracy : 0.9554
  f1       : 0.9552

BS Task - RoBERTa only
  log_loss : 0.0425
  roc_auc  : 0.9969
  accuracy : 0.9969
  f1       : 0.9969

BS Task - Combiner
  log_loss : 0.0303
  roc_auc  : 0.9995
  accuracy : 0.9969
  f1       : 0.9969
"""