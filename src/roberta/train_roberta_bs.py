from pathlib import Path

import pandas as pd

from src.roberta.roberta_model import RobertaModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGED_DIR = REPO_ROOT / "data_cleansing" / "merged"
MODEL_BASE = str(Path(__file__).resolve().parent / "roberta_bs_model")
TRAIN_OUTPUT_DIR = str(Path(__file__).resolve().parent / "roberta_bs_training")


def main() -> None:
    df = pd.read_csv(MERGED_DIR / "bullshit_training_all.csv", usecols=["text", "label1"])
    texts = df["text"].astype(str).to_numpy()
    y_bs = df["label1"].astype(int).to_numpy()

    model = RobertaModel(target_label="label1")
    model.train(texts, labels_1=y_bs, output_dir=TRAIN_OUTPUT_DIR)
    model.save_model(MODEL_BASE)

    print("Saved BS RoBERTa model to:", MODEL_BASE)


if __name__ == "__main__":
    main()
