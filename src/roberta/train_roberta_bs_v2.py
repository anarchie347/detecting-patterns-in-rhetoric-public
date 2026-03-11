from pathlib import Path
import os

import numpy as np
from sklearn.model_selection import train_test_split

from src.roberta.roberta_model_v2 import RobertaModelV2
from src.roberta.v2_split_utils import load_text_label_csv, split_train_cal_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
TRAINED_MODELS_DIR = REPO_ROOT / "trained_models"
MODEL_BASE = str(TRAINED_MODELS_DIR / "roberta_bs_model_v2")
TRAIN_OUTPUT_DIR = str(Path(__file__).resolve().parent / "roberta_bs_training_v2")

def main() -> None:
    max_token_len = int(os.environ.get("ROBERTA_MAX_TOKEN_LEN", "160"))
    batch_size = int(os.environ.get("ROBERTA_BATCH_SIZE", "2"))
    num_epochs = int(os.environ.get("ROBERTA_NUM_EPOCHS", "4"))
    val_size = float(os.environ.get("ROBERTA_VAL_SIZE", "0.05"))
    use_eval = os.environ.get("ROBERTA_USE_EVAL", "0").strip().lower() in {"1", "true", "yes"}
    optimizer = os.environ.get("ROBERTA_OPTIM", "adafactor")
    max_samples = int(os.environ.get("ROBERTA_MAX_SAMPLES", "0"))
    max_chars = int(os.environ.get("ROBERTA_TRAIN_MAX_CHARS", "1200"))
    split_seed = int(os.environ.get("V2_SPLIT_SEED", "42"))
    split_train = float(os.environ.get("V2_SPLIT_TRAIN_FRAC", "0.8"))
    split_cal = float(os.environ.get("V2_SPLIT_CAL_FRAC", "0.1"))
    split_meta = float(os.environ.get("V2_SPLIT_META_FRAC", "0.1"))

    texts_all, y_all = load_text_label_csv(
        MERGED_DIR / "bullshit_training_all.csv",
        text_col="text",
        label_col="label1",
        max_chars=max_chars,
    )
    splits = split_train_cal_meta(
        texts_all,
        y_all,
        train_frac=split_train,
        cal_frac=split_cal,
        meta_frac=split_meta,
        random_state=split_seed,
    )
    texts, y_bs = splits["train"]
    if max_samples > 0 and max_samples < len(texts):
        _, texts_sub, _, y_sub = train_test_split(
            texts,
            y_bs,
            test_size=max_samples,
            random_state=42,
            stratify=y_bs,
        )
        texts = texts_sub
        y_bs = y_sub

    print(
        "Training BS RoBERTa V2 with",
        f"max_token_len={max_token_len}, batch_size={batch_size},",
        f"num_epochs={num_epochs}, val_size={val_size},",
        f"use_eval={use_eval}, optimizer={optimizer},",
        f"max_samples={max_samples or 'all'}, max_chars={max_chars},",
        f"split=train({split_train:.2f})/cal({split_cal:.2f})/meta({split_meta:.2f}), seed={split_seed},",
        f"train_split_size={len(texts)}",
    )

    model = RobertaModelV2(target_label="label1", max_token_len=max_token_len)
    model.train(
        texts,
        labels_1=y_bs,
        output_dir=TRAIN_OUTPUT_DIR,
        batch_size=batch_size,
        num_epochs=num_epochs,
        val_size=val_size,
        use_eval_during_training=use_eval,
        optimizer=optimizer,
    )
    model.save_model(MODEL_BASE)

    print("Saved BS RoBERTa V2 model to:", MODEL_BASE)


if __name__ == "__main__":
    main()
