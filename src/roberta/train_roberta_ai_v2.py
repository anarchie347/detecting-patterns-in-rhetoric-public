from pathlib import Path
import os

import numpy as np
from sklearn.model_selection import train_test_split

from src.roberta.roberta_model_v2 import RobertaModelV2
from src.roberta.v2_split_utils import load_text_label_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_ai_model_v2")
TRAIN_OUTPUT_DIR = str(Path(__file__).resolve().parent / "roberta_ai_training_v2")

def main() -> None:
    max_token_len = int(os.environ.get("ROBERTA_MAX_TOKEN_LEN", "384"))
    batch_size = int(os.environ.get("ROBERTA_BATCH_SIZE", "2"))
    num_epochs = int(os.environ.get("ROBERTA_NUM_EPOCHS", "4"))
    val_size = float(os.environ.get("ROBERTA_VAL_SIZE", "0.05"))
    use_eval = os.environ.get("ROBERTA_USE_EVAL", "0").strip().lower() in {"1", "true", "yes"}
    optimizer = os.environ.get("ROBERTA_OPTIM", "adafactor")
    max_samples = int(os.environ.get("ROBERTA_MAX_SAMPLES", "0"))
    max_chars = int(os.environ.get("ROBERTA_TRAIN_MAX_CHARS", "4096"))

    texts, y_ai = load_text_label_csv(
        MERGED_DIR / "ai_training_all.csv",
        text_col="text",
        label_col="label0",
        max_chars=max_chars,
    )
    if max_samples > 0 and max_samples < len(texts):
        _, texts_sub, _, y_sub = train_test_split(
            texts,
            y_ai,
            test_size=max_samples,
            random_state=42,
            stratify=y_ai,
        )
        texts = texts_sub
        y_ai = y_sub

    print(
        "Training AI RoBERTa V2 with",
        f"max_token_len={max_token_len}, batch_size={batch_size},",
        f"num_epochs={num_epochs}, val_size={val_size},",
        f"use_eval={use_eval}, optimizer={optimizer},",
        f"max_samples={max_samples or 'all'}, max_chars={max_chars},",
        f"training_size={len(texts)}",
    )

    model = RobertaModelV2(target_label="label0", max_token_len=max_token_len)
    model.train(
        texts,
        labels_0=y_ai,
        output_dir=TRAIN_OUTPUT_DIR,
        batch_size=batch_size,
        num_epochs=num_epochs,
        val_size=val_size,
        use_eval_during_training=use_eval,
        optimizer=optimizer,
    )
    model.save_model(MODEL_BASE)

    print("Saved AI RoBERTa V2 model to:", MODEL_BASE)


if __name__ == "__main__":
    main()

"""
AI RoBERTa V2 (label0) metrics
  log_loss: 0.05122084254386494
  roc_auc : 0.998792410169448
  acc     : 0.989247311827957
  f1      : 0.9892857142857143
"""
