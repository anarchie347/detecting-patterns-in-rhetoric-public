import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.combine_output.combiner_both import load_combiner

EPS = 1e-6

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
COMBINER_MODEL_DIR = TRAINED_MODELS_DIR / "combiner_model"

ROBERTA_AI_DIR = REPO_ROOT / "src" / "roberta" / "roberta_ai_model_v2"
ROBERTA_BS_DIR = REPO_ROOT / "src" / "roberta" / "roberta_bs_model_v2"
XGB0_NAME = str((TRAINED_MODELS_DIR / "xgb_ai_model" / "xgb_ai_model"))
XGB1_NAME = str((TRAINED_MODELS_DIR / "xgb_bs_model" / "xgb_bs_model"))
META_PATH = str(COMBINER_MODEL_DIR / "combiner_v2.joblib")


def report_metrics(y_true: np.ndarray, p_pos: np.ndarray, title: str) -> None:
    y_true = y_true.astype(int)
    p_pos = np.clip(p_pos, EPS, 1 - EPS)
    y_pred = (p_pos >= 0.5).astype(int)

    print(f"\n{title}")
    print(f"  log_loss : {log_loss(y_true, p_pos):.4f}")
    print(f"  roc_auc  : {roc_auc_score(y_true, p_pos):.4f}")
    print(f"  accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  f1       : {f1_score(y_true, y_pred):.4f}")


def _require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    os.environ["COMBINER_ROBERTA_IMPL"] = "v2"

    _require_exists(MERGED_DIR / "ai_testing_all.csv", "AI test CSV")
    _require_exists(MERGED_DIR / "bullshit_testing_all.csv", "BS test CSV")
    _require_exists(ROBERTA_AI_DIR, "RoBERTa AI v2 dir")
    _require_exists(ROBERTA_BS_DIR, "RoBERTa BS v2 dir")
    _require_exists(Path(META_PATH), "combiner v2 artifact")

    df_ai = pd.read_csv(MERGED_DIR / "ai_testing_all.csv", usecols=["text", "label0"])
    texts_ai = df_ai["text"].astype(str).to_numpy()
    y0 = df_ai["label0"].astype(int).to_numpy()

    df_bs = pd.read_csv(MERGED_DIR / "bullshit_testing_all.csv", usecols=["text", "label1"])
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

    p0_xgb = combiner._xgb_predict(combiner.xgb0, texts_ai)
    p1_xgb = combiner._xgb_predict(combiner.xgb1, texts_bs)
    p0_rob = combiner.rob0.predict(texts_ai)
    p1_rob = combiner.rob1.predict(texts_bs)

    print(
        "\nCombiner config:"
        f" feature_mode={getattr(combiner, 'feature_mode', 'unknown')},"
        f" logit_eps={getattr(combiner, 'logit_eps', 'unknown')}"
    )

    report_metrics(y0, p0_xgb, "AI Task - XGBoost only (v2 eval)")
    report_metrics(y0, p0_rob, "AI Task - RoBERTa only (v2 eval)")
    report_metrics(y0, p0_comb, "AI Task - Combiner (v2 eval)")

    report_metrics(y1, p1_xgb, "BS Task - XGBoost only (v2 eval)")
    report_metrics(y1, p1_rob, "BS Task - RoBERTa only (v2 eval)")
    report_metrics(y1, p1_comb, "BS Task - Combiner (v2 eval)")


if __name__ == "__main__":
    main()


"""
AI Task - XGBoost only (v2 eval)
  log_loss : 0.2484
  roc_auc  : 0.9770
  accuracy : 0.8943
  f1       : 0.8841

AI Task - RoBERTa only (v2 eval)
  log_loss : 0.0512
  roc_auc  : 0.9988
  accuracy : 0.9892
  f1       : 0.9893

AI Task - Combiner (v2 eval)
  log_loss : 0.0650
  roc_auc  : 0.9981
  accuracy : 0.9910
  f1       : 0.9911

BS Task - XGBoost only (v2 eval)
  log_loss : 0.1094
  roc_auc  : 0.9945
  accuracy : 0.9554
  f1       : 0.9552

BS Task - RoBERTa only (v2 eval)
  log_loss : 0.0277
  roc_auc  : 0.9996
  accuracy : 0.9938
  f1       : 0.9938

BS Task - Combiner (v2 eval)
  log_loss : 0.0291
  roc_auc  : 0.9996
  accuracy : 0.9954
  f1       : 0.9954
"""