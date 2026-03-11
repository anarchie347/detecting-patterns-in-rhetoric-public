from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.combine_output.combiner_both import Combiner

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
COMBINER_MODEL_DIR = Path(__file__).resolve().parent / "combiner_model"
COMBINER_MODEL_DIR.mkdir(exist_ok=True)

# Expected pre-trained RoBERTa model directories.
ROBERTA_AI_DIR = REPO_ROOT / "src" / "roberta" / "roberta_ai_model"
ROBERTA_BS_DIR = REPO_ROOT / "src" / "roberta" / "roberta_bs_model"

META_PATH = str(COMBINER_MODEL_DIR / "combiner.joblib")


def main() -> None:
    df_ai = pd.read_csv(MERGED_DIR / "ai_training_all.csv")
    texts_ai = df_ai["text"].astype(str).to_numpy()
    y0 = df_ai["label0"].astype(int).to_numpy()

    df_bs = pd.read_csv(MERGED_DIR / "bullshit_training_all.csv")
    texts_bs = df_bs["text"].astype(str).to_numpy()
    y1 = df_bs["label1"].astype(int).to_numpy()

    # Mirror 80/10/10 split pattern separately for each head.
    ai_texts_tr, ai_texts_te, y0_tr, y0_te = train_test_split(
        texts_ai, y0, test_size=0.1, random_state=42, stratify=y0
    )
    ai_texts_tr, ai_texts_va, y0_tr, y0_va = train_test_split(
        ai_texts_tr, y0_tr, test_size=0.1111, random_state=42, stratify=y0_tr
    )
    bs_texts_tr, bs_texts_te, y1_tr, y1_te = train_test_split(
        texts_bs, y1, test_size=0.1, random_state=42, stratify=y1
    )
    bs_texts_tr, bs_texts_va, y1_tr, y1_va = train_test_split(
        bs_texts_tr, y1_tr, test_size=0.1111, random_state=42, stratify=y1_tr
    )

    # Fit meta models on each head's held-out pool (val + test).
    ai_texts_comb = np.concatenate([ai_texts_va, ai_texts_te])
    y0_comb = np.concatenate([y0_va, y0_te])
    bs_texts_comb = np.concatenate([bs_texts_va, bs_texts_te])
    y1_comb = np.concatenate([y1_va, y1_te])

    combiner = Combiner()

    if not ROBERTA_AI_DIR.exists() or not ROBERTA_BS_DIR.exists():
        raise FileNotFoundError(
            "Missing RoBERTa model directories. Train both first:\n"
            f"- {ROBERTA_AI_DIR}\n"
            f"- {ROBERTA_BS_DIR}"
        )

    # Load pre-trained RoBERTa models.
    combiner.rob0.load_model("unused", model_dir=str(ROBERTA_AI_DIR))
    combiner.rob1.load_model("unused", model_dir=str(ROBERTA_BS_DIR))

    # Train base XGBoost models on strictly separate task-specific data.
    combiner.xgb0.train_new_model(ai_texts_tr, y0_tr.astype(np.uint8))
    combiner.xgb1.train_new_model(bs_texts_tr, y1_tr.astype(np.uint8))

    # Train and save only the meta-learners artifact.
    combiner.fit_meta_separate(ai_texts_comb, y0_comb, bs_texts_comb, y1_comb)
    combiner.save(META_PATH)

    print("Saved combiner meta artifact:")
    print(" ", META_PATH)


if __name__ == "__main__":
    main()
