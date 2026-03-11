from pathlib import Path

import pandas as pd

from src.roberta.roberta_model import RobertaModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_ai_model")
TRAIN_OUTPUT_DIR = str(Path(__file__).resolve().parent / "roberta_ai_training")


def main() -> None:
    df = pd.read_csv(MERGED_DIR / "ai_training_all.csv", usecols=["text", "label0"])
    texts = df["text"].astype(str).to_numpy()
    y_ai = df["label0"].astype(int).to_numpy()

    model = RobertaModel(target_label="label0")
    model.train(texts, labels_0=y_ai, output_dir=TRAIN_OUTPUT_DIR)
    model.save_model(MODEL_BASE)

    print("Saved AI RoBERTa model to:", MODEL_BASE)


if __name__ == "__main__":
    main()
